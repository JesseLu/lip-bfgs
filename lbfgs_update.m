function [delta, M, W, h] = lbfgs_update(x, g, n_max, h)
% [DELTA, M, W, HIST] = LBFGS_UPDATE(X, G, N_MAX, HIST)
% 
% Description
%     Update the L-BFGS approximation to the Hessian of a function.
% 
%     The L-BFGS approximation to a Hessian, H, is given by
%         
%         H ~ DELTA * I - W * M * W*T.
% 
%     This is known as the compact, or outer product, representation of the 
%     quasi-Newton matrix (see Reference).
%
% 
% Inputs
%     X: Vector.
%         Current value of X, the optimization variable.
% 
%     G: Vector.
%         Current value of the gradient of the function to be approximated.
% 
%     N_MAX: Positive integer.
%         The maximum number of previous (X, G) pairs to retain in the
%         approximation. Recommended values of N_MAX are 3 and 5.
% 
%     HIST: Structure.
%         Structure containing the previous (X, G) pairs of the function, 
%         as well as associated computed values.
% 
%         To initiate a restart, use HIST = [].
%
% 
% Outputs
%     DELTA: Positive scalar.
%         See below.
% 
%     M, W: Matrices.
%         DELTA, M, and W form the quasi-Newton (L-BFGS) approximation of the
%         Hessian, as outlined above.
% 
%     HIST: Structure.
%         Contains the updated history of previous (X, G) pairs. Use HIST as an
%         input to the subsequent L-BFGS update.
% 
% 
% Reference
%     Chapter 7.2, Nocedal and Wright, Numerical Optimization (Cambridge 2004).


    %
    % Check if we need to restart.
    % A restart (or start) is initiated if there is no history.
    %

if isempty(h) 
    % Components of the quasi-Newton approximation.
    delta = 1; % For restart, we simply guess a scaling value of 1.
    W = zeros(length(x),0);
    M = zeros(0);

    % Initialize the history structure.
    h.n = 0; % Index, tells us how "full" S and Y are.
    h.S = zeros(length(x), n_max); % Corresponds to S in the reference.
    h.Y = zeros(length(x), n_max); % Corresponds to Y in the reference.
    h.x_prev = x; % Used to calculate s for next iteration.
    h.g_prev = g; % Used to calculate y for next iteration.
    return
end


    %
    % Calculate currest values of s and y. 
    %

s = x - h.x_prev;
y = g - h.g_prev;


    %
    % Update history.
    %

h.x_prev = x;
h.g_prev = g;

if h.n < n_max % History not full.
    n = h.n + 1;
else % History full, delete oldest entry.
    n = n_max;
    h.S(:, 1:n_max-1) = h.S(:, 2:n_max);
    h.Y(:, 1:n_max-1) = h.Y(:, 2:n_max);
    h.d(:, 1:n_max-1) = h.d(2:n_max);
    h.L(1:n_max-1, 1:n_max-1) = h.L(2:n_max, 2:n_max);
    h.S_dot_S(1:n_max-1, 1:n_max-1) = h.S_dot_S(2:n_max, 2:n_max);
end
h.n = n;

% Check curvature condition.
if ((s' * y) <= 0)
    error('Curvature condition broken (s dot y = %e)!', (s' * y));
    % TODO (safeguard): Replace y (damped update) when condition is broken.
end

% Insert new values of s and y into the history.
h.S(:,n) = s;
h.Y(:,n) = y;


    %
    % Calculate components needed to form delta, M, and W.
    %

% These dot products should be the bulk of the processing for this function.
s_dot_y = s' * h.Y(:, 1:n);
s_dot_s = s' * h.S(:, 1:n);

% Update components used to form delta, M, and W.
h.S_dot_S(n, 1:n) = s_dot_s; % Inner product S' * S.
h.S_dot_S(1:n-1, n) = s_dot_s(1:n-1);

h.d(n) = s_dot_y(n); % Vector containing diagonal elements of a diagonal matrix.

h.L(n, 1:n) = [s_dot_y(1:n-1), 0]; % Lower-diagonal matrix.


    %
    % Form delta, M, and W. 
    % The compact representation of the L-BFGS approximation.
    %

delta = norm(y)^2 / s_dot_y(n); % Scaling factor.

% Use matrix inverse, since M is a square matrix of size (2*n_max) x (2*n_max).
M = inv([delta*h.S_dot_S, h.L; h.L', diag(-h.d)]);

W = [delta*h.S(:, 1:n), h.Y(:, 1:n)];

