function [] = test_lbfgs_ip(n, p, err_tol)
% Test lbfgs_ip

randn('state', 1);


    %
    % Formulate problem.
    %

% Function to minimize (quadratic, convex).
A = spdiags((randn(n, 25)), -12:12, n, n);  % Use sparse matrix to speed things up.
A = A' * A;
b = 1*randn(n, 1);
fun.f = @(x) 0.5 * x' * A * x - x' * b;
fun.g = @(x) A * x - b;
fun.H = @(x) A;

% fun.f_cvx = @(x) norm(A * x - b);

lip_bfgs(fun, zeros(n, 1), -ones(n, 1), ones(n, 1));
 
