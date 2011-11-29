function [x, hist] = lip_bfgs(fun, x, l, u, n_max, mu_progress, alpha, beta)

n = length(x);

f = @(x, mu) my_fun(fun, x, mu, l, u);
g = @(x, mu) fun.g(x) + mu * ((u - x).^-1 - (x - l).^-1);
H = @(x, mu) fun.H(x) + mu * spdiags((u - x).^-2 + (x - l).^-2, 0, n, n);
phi = @(x, mu) norm(g(x, mu)) / sqrt(n);

h = [];
cnt = 0;
start_time = tic;
prev_time = 0;
for mu = mu_progress
    while true
        % Direct solve of the full Hessian.
        p = H(x, mu) \ -g(x, mu);

        % Determine p using quasi-newton approximation.
        [delta, M, W, h, is_damp, is_restart] = ...
            lbfgs_update(x, g(x, 0), n_max, h);
        p = arrow_solve(delta + mu * ((u - x).^-2 + (x - l).^-2), ...
            zeros(0, n), -W * M, W, -g(x, mu));

        % Perform backtracking line search.
        [x, t] = backtrack(@(x) f(x, mu), @(x) g(x, mu), x, p, alpha, beta, 1);

        if isnan(t) % If line search failed, restart quasi-Newton approximation.
            h = [];
        end

        % Store algorithm progress.
        cnt = cnt + 1; % Counter variable.
        hist.fval(cnt) = f(x, 0);
        hist.err(cnt) = phi(x, mu);
        hist.mu(cnt) = mu;
        hist.t(cnt) = t;
        hist.is_damp(cnt) = is_damp;
        hist.is_restart(cnt) = is_restart;

        % Output progress information
        if (toc(start_time) - prev_time > 1) % Do this every ~1 second. 
            if (prev_time == 0)
                fprintf('iter#       fval           mu          err        step len\n');
                fprintf('-----    ----------    ---------    ---------    ---------\n');
            end
            prev_time = toc(start_time);
            fprintf('%5d    %+1.3e    %1.3e    %1.3e    %1.3e\n', ...
                cnt, hist.fval(cnt), hist.mu(cnt), hist.err(cnt), hist.t(cnt));

            % Plot information.
            subplot(1, 3, 1); plot(hist.fval, '.-');
            subplot(1, 3, [2 3]); semilogy([hist.mu; hist.err; hist.t]', '.-');
            hold on; semilogy(hist.t .* hist.is_damp, 'k.'); hold off;
            hold on; semilogy(hist.t .* hist.is_restart, 'mo'); hold off;
            drawnow;
        end

        if (phi(x, mu)*1 < mu)
            break
        end
    end
end

    %
    % Print out results.
    %

fprintf('%5d    %+1.3e    %1.3e    %1.3e    %1.3e    (%1.1f secs)\n', cnt, ...
    hist.fval(cnt), hist.mu(cnt), hist.err(cnt), hist.t(cnt), toc(start_time));
subplot(1, 3, 1); plot(hist.fval, '.-');
subplot(1, 3, [2 3]); semilogy([hist.mu; hist.err; hist.t]', '.-');
            hold on; semilogy(hist.t .* hist.is_damp, 'k.'); hold off;
            hold on; semilogy(hist.t .* hist.is_restart, 'mo'); hold off;



function [x, t] = backtrack(f, g, x, p, alpha, beta, t)

while f(x) + t * alpha * g(x) <= f(x + t * p)
    t = t * beta;
    if (t <= 1e-6)
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
    
