R = 5;

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
    
    [probstruct, history] = infbench_run('vbmc18', 'lumpy', 2, [],'vbmc', r, options);
    
    prefix = sprintf("vbmc-lumpy-run-%02d-", r);
    algo_out = history{1,1}.Output;
    log_evidence = history{1,1}.lnZpost_true;
    stats = algo_out.stats;
    csvwrite(strcat(prefix, "elbo.csv"), stats.elbo);       % mean ELBO
    csvwrite(strcat(prefix, "gsKL.csv"), algo_out.gsKL);    % Gaussian symmetrized KL divergence
    csvwrite(strcat(prefix, "kl.csv"), log_evidence - stats.elbo);
    csvwrite(strcat(prefix, "f_count.csv"), stats.funccount);
end