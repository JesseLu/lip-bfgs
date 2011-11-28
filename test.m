function [] = test_lbfgs_ip(n, p, err_tol)
% Test lbfgs_ip

randn('state', 1);


    %
    % Formulate problem.
    %

% Function to minimize (quadratic, convex).
A = spdiags((randn(n, 15)), -7:7, n, n);  % Use sparse matrix to speed things up.
Ahat = A' * A;
% Ahat = eye(n);
b = 1*randn(n, 1);
fun.f = @(x) 0.5 * x' * Ahat * x - x' * b;
fun.g = @(x) Ahat * x - b;
fun.H = @(x) Ahat;

% fun.f_cvx = @(x) norm(A * x - b);

lip_bfgs(fun, zeros(n, 1), -ones(n, 1), ones(n, 1));
 
