function [t] = backtrack_linesearch(f, t, alpha, beta);
% Backtracking line search on one-dimensional function f.
% t = 1;
f0 = f(0);
y = []; % For debugging purposes.
x = [];
while f(t) > (1 - alpha * t) * f(0)
    y(end+1) = f(t);
    x(end+1) = t;
    t = beta * t;
    if (t <= 1e-6) % Just try to get some improvement.
        warning('Setting alpha to 0');
        alpha = 0;
    end
    if (t <= eps) % Not a descent direction.
        semilogx(x, y - f0, '.-');
        drawnow
        error('Backtracking line-search failed.');
        break
    end
end

