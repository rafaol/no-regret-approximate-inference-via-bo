D = 2;                          % We consider a 2-D problem
R = 5;

 
% We define now a prior over the parameters (for simplicity, independent
% Gaussian prior on each variable, but you could do whatever).

prior_mu = zeros(1,D);
prior_var = 3^2*ones(1,D);

% So our log joint (that is, unnormalized log posterior density), is:
fun = @(x) infbench_lumpy(x, []);

% We assume an unconstrained domain for the model parameters, and finite 
% plausible bounds which should denote a region of high posterior 
% probability mass. Not knowing better, we use mean +/- 1 SD of the prior 
% (that is, the top ~68% prior credible interval) to set plausible bounds.

LB = -Inf(1,D);                            % Lower bounds
UB = Inf(1,D);                             % Upper bounds
PLB = prior_mu - sqrt(prior_var);          % Plausible lower bounds
PUB = prior_mu + sqrt(prior_var);          % Plausible upper bounds

% Analogously, you could set the plausible bounds using the quantiles:
% PLB = norminv(0.1587,prior_mu,sqrt(prior_var));
% PUB = norminv(0.8413,prior_mu,sqrt(prior_var));

% As a starting point, we use the mean of the prior:
x0 = prior_mu;

% Alternatively, we could have used a sample from inside the plausible box:
% x0 = PLB + rand(1,D).*(PUB - PLB);

for r=1:R
    fprintf("############################\n");
    fprintf("Run %02d out of %02d\n", r, R);
    fprintf("############################\n\n");
    % For now, we use default options for the inference:
    options = struct('SpeedTests', false);
    options.MaxFunEvals = 100;
    options.MinFunEvals = options.MaxFunEvals;
    % options.NSgpMax = 0;
    options.Warmup = 'off';
    % options.GPTrainNinit = 0;
    % options.MaxIter = 20;
    % options.MinIter = options.MaxIter;
    options.WarpRotoScaling = 'no';
    
    % Run VBMC, which returns the variational posterior VP, the lower bound 
    % on the log model evidence ELBO, and its uncertainty ELBO_SD.
    [vp,elbo,elbo_sd,exitflag,output,optimState,stats] = vbmc(fun,x0,LB,UB,PLB,PUB,options);
    
    prefix = sprintf("vbmc-lumpy-run-%02d-", r);
    csvwrite(strcat(prefix, "elbo.csv"), stats.elbo);       % mean ELBO
    csvwrite(strcat(prefix, "gsKL.csv"), stats.sKL_true);    % Gaussian symmetrized KL divergence
    csvwrite(strcat(prefix, "f_count.csv"), stats.funccount);
end