function [] = test_lbfgs_ip(n, p, err_tol)
% Test lbfgs_ip

randn('state', 1);


    %
    % Formulate problem.
    %

% Function to minimize (quadratic, convex).
A = spdiags((randn(n, 3)), -1:1, n, n);  % Use sparse matrix to speed things up.
% A = randn(n);
% A = A' * A;
b = 1*randn(n, 1);
fun.f = @(x) x' * A * x - x' * b;
fun.g = @(x) (A + A') * x - b;
fun.H = @(x) (A + A');

% fun.f_cvx = @(x) norm(A * x - b);

lip_bfgs(fun, zeros(n, 1), -1e0 * ones(n, 1), 1e0 * ones(n, 1));
 
