function [] = test_lbfgs_ip(n, p, err_tol)
% Test lbfgs_ip

randn('state', 1);


    %
    % Formulate problem.
    %

% Function to minimize (quadratic, convex).
A = -10*speye(n) * spdiags((randn(n, 7)), -3:3, n, n);  % Use sparse matrix to speed things up.
A = spdiags((randn(n, 15)), -7:7, n, n);  % Use sparse matrix to speed things up.
% A = randn(n);
Ahat = A' * A;
% A = randn(n);
b = 1*randn(n, 1);
% fun.f = @(x) x' * (A * x - b);
fun.g = @(x) Ahat * x - b;
% fun.h = @(x) A' * A;
tic
Ahat \ b; toc

fun.f_cvx = @(x) norm(A * x - b);

% Equality constraint.
A_eq = randn(p, n);
b_eq = randn(p, 1);

% Inequality constraint.
A_in = speye(n);
l = -1*ones(n, 1);
u = 1*ones(n, 1);

subplot 111;
lip_bfgs(fun.g, zeros(n, 1), l, u, A_eq, b_eq, ...
    1, 10, 0.995, 10, 1e-4, 0.5, err_tol, 1e-15, 5);
 
