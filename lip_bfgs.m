function [] = lip_bfgs(fun, x, l, u)

n = length(x);

f = @(x, mu) my_fun(fun, x, mu, l, u);
g = @(x, mu) fun.g(x) + mu * ((u - x).^-1 - (x - l).^-1);
H = @(x, mu) fun.H(x) + mu * spdiags((u - x).^-2 + (x - l).^-2, 0, n, n);
phi = @(x, mu) norm(g(x, mu)) / sqrt(n);

h = [];
n_max = 10;
cnt = 0;
for mu = 10.^[4:-1:-4]
    while true
        % Direct solve of the full Hessian.
        p = H(x, mu) \ -g(x, mu);

        % Determine p using quasi-newton approximation.
        [delta, M, W, h] = lbfgs_update(x, g(x, 0), n_max, h);
        p = arrow_solve(delta + mu * ((u - x).^-2 + (x - l).^-2), ...
            zeros(0, n), -W * M, W, -g(x, mu));

%         A = -W * M * W' + ...
%             spdiags(delta + mu * ((u - x).^-2 + (x - l).^-2), 0, n, n);
        % min(real(eig(full(A))))

        [x, t] = backtrack(@(x) f(x, mu), @(x) g(x, mu), x, p); %-1e-0 * g(x, mu));
%         if isnan(t)
%             g0 = g(x, mu);
%             f0 = f(x, mu);
%             t = [-1 : 1e-3 : 1] * 1e-3;
%             for k = 1 : length(t)
%                 y(k) = f(x - t(k) * g0, mu) - f0;
%             end
%             plot(t, [y; -1e-1 * t]', '.-');
%             return
%         end
        if isnan(t)
            h = [];
            p = -g(x, mu);
            [x, t] = backtrack(@(x) f(x, mu), @(x) g(x, mu), x, p);
            if isnan(t)
                error('hmm');
            end
        end

        cnt = cnt + 1;
        fprintf('%d: %e %e (%e) [%e]\n', cnt, f(x, mu), phi(x, mu), t, mu);
        if (phi(x, mu) < 1e-3)
            break
        end
    end
    fprintf('%e [%e]\n\n', phi(x, mu), mu);
end
% % Fraction-to-boundary rule (for inequality constraint).
% my_pos = @(z) (z > 0) .* z + (z <= 0) * 1; % If negative, set to 1.
% f2b_rule = @(pz, z) min([1; my_pos(-tau*z./pz)]);

function [x, t] = backtrack(f, g, x, p)
alpha = 1e-4;
beta = 0.5;
t = 1;

while f(x) + t * alpha * g(x) <= f(x + t * p)
    t = t * beta;
    if (t <= 1e-15)
        warning('Backtrack fail.');
        t = nan;
        return
    end
    % fprintf('%e\n', f(x + t * p) - f(x));
end
x = x + t * p;


function [f] = my_fun(fun, x, mu, l, u)
if any(x <= l) | any(x >= u)
    f = Inf;
    % error('Infeasible.');
else
    f = fun.f(x) - mu * sum(log(u - x) + log(x - l));
end
    
