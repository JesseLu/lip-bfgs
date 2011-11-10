function [t] = backtrack_linesearch(f, t, alpha, beta, t_min);
% Backtracking line search on one-dimensional function f.
% t = 1;
f0 = f(0);
while f(t) > (1 - alpha * t) * f(0)
    t = beta * t;
    if (t <= t_min) % Just try to get some improvement.
        error('Backtracking line search failed.');
    end
end

