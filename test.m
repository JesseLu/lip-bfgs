function [] = test_lbfgs_ip(n, p, err_tol)
% Test lbfgs_ip

randn('state', 1);


    %
    % Formulate problem.
    %

% Function to minimize (quadratic, convex).
A = spdiags(randn(n, 5), -2:2, n, n);  % Use sparse matrix to speed things up.
% A = randn(n);
b = randn(n, 1);
fun.f = @(x) 0.5 * norm(A * x - b)^2;
fun.g = @(x) A' * (A * x - b);
fun.h = @(x) A' * A;

fun.f_cvx = @(x) norm(A * x - b);

% Equality constraint.
A_eq = randn(p, n);
b_eq = randn(p, 1);

% Inequality constraint.
A_in = speye(n);
l = zeros(n, 1);
u = ones(n, 1);

subplot 111;
lip_bfgs(fun, 0.5 * ones(n, 1), l, u, A_eq, b_eq, ...
    1, 0.1, 0.995, 1, 1e-4, 0.5, err_tol, 1e-6);
