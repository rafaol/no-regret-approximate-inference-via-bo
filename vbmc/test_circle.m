D = 2;                          % We consider a 2-D problem
R = 5;

llfun = @circle_likelihood;

% LLFUN would be the function handle to the log likelihood of your model.
 
% We define now a prior over the parameters (for simplicity, independent
% Gaussian prior on each variable, but you could do whatever).

prior_mu = zeros(1,D);
prior_var = ones(1,D);
lpriorfun = @(x) ...
    -0.5*sum((x-prior_mu).^2./prior_var,2) ...
    -0.5*log(prod(2*pi*prior_var));

% So our log joint (that is, unnormalized log posterior density), is:
fun = @(x) llfun(x) + lpriorfun(x);

LB = -Inf(1,D);                            % Lower bounds
UB = Inf(1,D);                             % Upper bounds
PLB = prior_mu - sqrt(prior_var);          % Plausible lower bounds
PUB = prior_mu + sqrt(prior_var);          % Plausible upper bounds

% As a starting point, we use the mean of the prior:
x0 = prior_mu;

% Alternatively, we could have used a sample from inside the plausible box:
% x0 = PLB + rand(1,D).*(PUB - PLB);

for r = 1:R
    fprintf("############################\n");
    fprintf("Run %02d out of %02d\n", r, R);
    fprintf("############################\n\n");
    options = vbmc('defaults');
    options.MaxFunEvals = 100;
    options.MinFunEvals = options.MaxFunEvals;
    options.WarpRotoScaling = 'no';
    options.TrueMean = zeros(1, 2);
    options.TrueCov = [1.0776, 0.0100; 0.0100, 1.1262];

    % Run VBMC, which returns the variational posterior VP, the lower bound 
    % on the log model evidence ELBO, and its uncertainty ELBO_SD.
    [vp,elbo,elbo_sd,exitflag,output,optimState,stats] = vbmc(fun,x0,LB,UB,PLB,PUB,options);
    
    % Save results
    prefix = sprintf("vbmc-circle-run-%02d-", r);
    csvwrite(strcat(prefix, "elbo.csv"), stats.elbo);           % mean ELBO
    csvwrite(strcat(prefix, "sKL_true.csv"), stats.sKL_true);   % Gaussian symmetrized KL divergence
    csvwrite(strcat(prefix, "f_count.csv"), stats.funccount);
end
