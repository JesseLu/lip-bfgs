function [] = lbfgs_ip(fun, x, l, u, A, b, ...
                        mu_0, sigma, tau, eta, alpha, beta, err_tol, t_min)
% L-BFGS Interior-Point algorithm.
% Assumes A is skinny and has linearly independent columns.

    %
    % Set up variables and helper functions.
    %

% Choose initial values of variables
s0 = x - l; % Slack variable for lower bound.
z0 = ones(length(l), 1); % Dual variable for lower bound.
s1 = u - x; % Slack variable for upper bound.
z1 = ones(length(u), 1); % Dual variable for upper bound.
y = zeros(length(b), 1); % Dual variable for equality constraint.

% RHS of KKT equation.
kkt_res = @(x, s0, s1, y, z0, z1, mu) cat(1, ...
                        fun.g(x) - A' * y ...
                        - z0 + (z0 ./ s0) .* (x - l) - mu * s0.^-1 ...
                        + z1 - (z1 ./ s1) .* (u - x) + mu * s1.^-1, ...
                        A * x - b);

% Merit function.
phi = @(x, s0, s1, y, z0, z1, mu, alpha_prim, alpha_dual, p, t) ...
    norm(kkt_res(   x + t * alpha_prim * p.x, ...
                    s0 + t * alpha_prim * p.s0, ...
                    s1 + t * alpha_prim * p.s1, ...
                    y + 1 * alpha_dual * p.y, ...
                    z0 + 1 * alpha_dual * p.z0, ...
                    z1 + 1 * alpha_dual * p.z1, mu));

% Error function.
err = @(x, s0, s1, y, z0, z1, mu) max(cat(1, ...
                        norm(fun.g(x) - A' * y - z0 + z1), ...
                        norm(s0 .* z0 - mu), ...
                        norm(s1 .* z1 - mu), ...
                        norm(A * x - b), ...
                        norm(x - l - s0), ...
                        norm(u - x - s1)));

% Helper to calculate different components of p
calc_p_xy = @(p) struct(    'x', p(1:length(x)), ...
                            'y', -p(length(x) + [1:length(y)]));
calc_p_s0 = @(p, x, s0) p.x + (x - l) - s0;
calc_p_s1 = @(p, x, s1) -p.x + (u - x) - s1;
calc_p_z0 = @(p, x, s0, z0, mu) -(z0 ./ s0) .* (p.x + (x - l)) + mu * s0.^-1;
calc_p_z1 = @(p, x, s1, z1, mu) (z1 ./ s1) .* (p.x - (u - x)) + mu * s1.^-1;

% Fraction-to-boundary rule (for inequality constraint).
my_pos = @(z) (z > 0) .* z + (z <= 0) * 1; % If negative, set to 1.
f2b_rule = @(pz, z) min([1; my_pos(-tau*z./pz)]);


    %
    % Optimize!
    %

hist.err(1) = err(x, s0, s1, y, z0, z1, 0);
hist.t(1) = nan;
n_max = 5;
h = [];
tic
mu = mu_0;

while hist.err(end) > err_tol

    % L-BFGS approximation of Hessian function.
    [delta, M, W, h] = lbfgs_update(x, fun.g(x), n_max, h); 
    W = [W; zeros(size(A, 1), size(W, 2))];

    % Obtain search direction (p).
    p = arrow_solve(delta + z0./s0 + z1./s1, A, -W*M, W, ...
        -kkt_res(x, s0, s1, y, z0, z1, mu));

    % Split up p into various components.
    p = calc_p_xy(p);
    p.s0 = calc_p_s0(p, x, s0);
    p.s1 = calc_p_s1(p, x, s1);
    p.z0 = calc_p_z0(p, x, s0, z0, mu);
    p.z1 = calc_p_z1(p, x, s1, z1, mu);

    % Compute alpha using the fraction-to-boundary rule.
    alpha_prim = f2b_rule([p.s0; p.s1], [s0; s1]);
    alpha_dual = f2b_rule([p.z0; p.z1], [z0; z1]);

    % Perform a backtracking (Armijo) line search.
    t = backtrack_linesearch(@(t) ...
        phi(x, s0, s1, y, z0, z1, mu, alpha_prim, alpha_dual, p, t), ...
        alpha_prim, alpha, beta, t_min);

    % Update variables.
    x = x + t * alpha_prim * p.x;
    s0 = s0 + t * alpha_prim * p.s0;
    s1 = s1 + t * alpha_prim * p.s1;
    y = y + 1 * alpha_dual * p.y;
    z0 = z0 + 1 * alpha_dual * p.z0;
    z1 = z1 + 1 * alpha_dual * p.z1;

    % Calculate error.
    hist.err(end+1) = err(x, s0, s1, y, z0, z1, 0);
    hist.t(end+1) = t;

    % Update mu.
    if ((hist.err(end) / (length(x)/eta)) <= mu)
        mu = hist.err(end) / (length(x)/eta);
    end
end
time0 = toc;

% Plot results.
semilogy(0:length(hist.err)-1, [hist.err; hist.t]', '.-');
xlabel('Error in KKT equations');
ylabel('iterations');
title('Interior Primal-Dual Full Newton Step Convergence');
legend({'global error', 'step size'}, -1);
drawnow

fprintf('Time: %1.2fs, Final error: %e.\n', time0, hist.err(end));


