function [] = lip_bfgs(grad, x, l, u, A, b, ...
                        mu_0, mu_m, tau, eta, alpha, beta, err_tol, t_min, n_max)
% L-BFGS Interior-Point algorithm.
% Assumes A is fat and has linearly independent rows.

    %
    % Set up variables and helper functions.
    %

% Choose initial values of variables
% TODO: change initial s to allow for infeasible start.
s0 = x - l; % Slack variable for lower bound.
z0 = zeros(length(l), 1); % Dual variable for lower bound.
s1 = u - x; % Slack variable for upper bound.
z1 = zeros(length(u), 1); % Dual variable for upper bound.
y = zeros(length(b), 1); % Dual variable for equality constraint.

n = length(x);

% RHS of KKT equation.
kkt_res = @(g, x, s0, s1, y, z0, z1, mu) cat(1, ...
                        g - A' * y ...
                        - z0 + (z0 ./ s0) .* (x - l) - mu * s0.^-1 ...
                        + z1 - (z1 ./ s1) .* (u - x) + mu * s1.^-1, ...
                        A * x - b);
% Error function.
err0 = @(g, x, s0, s1, y, z0, z1, mu) 1/sqrt(n) * max(cat(1, ...
                        norm(g - A' * y - z0 + z1), ...
                        norm(s0 .* z0 - mu), ...
                        norm(s1 .* z1 - mu), ...
                        norm(A * x - b), ...
                        norm(x - l - s0), ...
                        norm(u - x - s1)));
% 
% % Merit function.
% phi = @(g, x, s0, s1, y, z0, z1, mu, alpha_prim, alpha_dual, p, t) ...
%     (err(   g, ...
%                     x + t * alpha_prim * p.x, ...
%                     s0 + t * alpha_prim * p.s0, ...
%                     s1 + t * alpha_prim * p.s1, ...
%                     y + 1 * alpha_dual * p.y, ...
%                     z0 + 1 * alpha_dual * p.z0, ...
%                     z1 + 1 * alpha_dual * p.z1, mu));

err = @(g, x, s0, s1, y, z0, z1, mu) ...
    norm(kkt_res(g, x, s0, s1, y, z0, z1, mu));
phi = @(g, x, s0, s1, y, z0, z1, mu, alpha_prim, alpha_dual, p, t) ...
    norm(kkt_res(   g, ...
                    x + t * alpha_prim * p.x, ...
                    s0 + t * alpha_prim * p.s0, ...
                    s1 + t * alpha_prim * p.s1, ...
                    y + 1 * alpha_dual * p.y, ...
                    z0 + 1 * alpha_dual * p.z0, ...
                    z1 + 1 * alpha_dual * p.z1, mu));

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
g = grad(x);
hist.err(1) = err(g, x, s0, s1, y, z0, z1, 0);
hist.t(1) = nan;
hist.grad_evals(1) = 1;
hist.search_fail(1) = false;
mu = hist.err / eta;
hist.err_mu(1) = err(g, x, s0, s1, y, z0, z1, 0);
hist.mu = mu;
h = [];

start_time = tic;
t_disp = 0.2;
cnt_display = 0;

while hist.err(end) > err_tol
    % while hist.err_mu(end) >= (mu * eta)

    % L-BFGS approximation of Hessian function.
    [delta, M, W, h] = sr1_update(x, grad(x), n_max, h); 
    % [delta, M, W, h] = lbfgs_update(x, grad(x), n_max, h); 
    W = [W; zeros(size(A, 1), size(W, 2))];

    % Obtain search direction (p).
    p = arrow_solve(delta + z0./s0 + z1./s1, A, -W*M, W, ...
        -kkt_res(g, x, s0, s1, y, z0, z1, mu));

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
        % pause
    [g, t, hist.grad_evals(end+1), hist.search_fail(end+1)] = my_backtrack_linesearch(...
        @(t) grad(x + t * alpha_prim * p.x), ...
        @(g, t) ...
        phi(g, x, s0, s1, y, z0, z1, mu, alpha_prim, alpha_dual, p, t), ...
        g, ...
        1.0, alpha, beta, t_min);
    if (hist.search_fail(end) == true) % Restart BFGS.
        h = [];
        % fprintf('Iteration %d\n', length(hist.err));
        t = 0;
    else
        % Update variables.
        x = x + t * alpha_prim * p.x;
        s0 = s0 + t * alpha_prim * p.s0;
        s1 = s1 + t * alpha_prim * p.s1;
        y = y + 1 * alpha_dual * p.y;
        z0 = z0 + 1 * alpha_dual * p.z0;
        z1 = z1 + 1 * alpha_dual * p.z1;
    end


    % Calculate error.
    hist.err(end+1) = err0(g, x, s0, s1, y, z0, z1, 0);
    hist.err_mu(end+1) = err(g, x, s0, s1, y, z0, z1, 0);
    hist.t(end+1) = t;
    hist.mu(end+1) = mu;

    % Update mu.
    if ((hist.err(end) / eta) <= mu)
        mu = hist.err(end) / eta;
    end

    % Output progress.
    if (toc(start_time) - cnt_display * t_disp > t_disp)
        if (cnt_display == 0)
            fprintf('Iter#    Gevals        Error        Errmu        Mu\n');
            fprintf('-----    ------        -----        -----        --\n');
        end

        fprintf('%5d    %6d   %1.4e   %1.4e   %1.1e\n', ...
            length(hist.err) - 1, sum(hist.grad_evals), hist.err(end), ...
            hist.err_mu(end), hist.mu(end));
        cnt_display = cnt_display + 1;
        subplot 211; semilogy(cumsum(hist.grad_evals), [hist.err; hist.err_mu; hist.t]', '.-');
        subplot 212; plot(x, '.-');
        drawnow
    end
%     end
%     mu = mu / mu_m;
    % pause
end
run_time = toc(start_time);

        subplot 211; semilogy(cumsum(hist.grad_evals), [hist.err; hist.err_mu; hist.t]', '.-');
        subplot 212; plot(x, '.-');
        drawnow
% % Plot results.
% semilogy(0:length(hist.err)-1, [hist.err; hist.t]', '.-');
% xlabel('Error in KKT equations');
% ylabel('iterations');
% title('Interior Primal-Dual Full Newton Step Convergence');
% legend({'global error', 'step size'}, -1);
% drawnow

fprintf('Time: %1.2fs, Final error: %e.\n', run_time, hist.err(end));


function [g, t, grad_evals, search_fail] = ...
    my_backtrack_linesearch(grad, f, g0, t, alpha, beta, t_min);
% Backtracking line search on one-dimensional function f.
% t = 1;
f0 = f(g0, 0);
g = grad(t);
grad_evals = 1;
while f(g, t) > (1 - alpha * t) * f0
    t = beta * t;
    g = grad(t);
    grad_evals = grad_evals + 1;
    if (t <= t_min) 
        % warning('Backtracking line search failed.');
        t = nan;
        search_fail = true;
        return
    end
end
search_fail = false;
