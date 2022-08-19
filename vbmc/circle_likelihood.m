function [y,s] = circle_likelihood(x,sigma)
%CIRCLE_LIKELIHOOD

if nargin < 2 || isempty(sigma); sigma = 0; end

% Circular likelihood
radius = 1.5;
lengthscale = 0.25;
y = -((sqrt(sum(x.^2,2))-radius)/lengthscale).^2;


% Noisy test
if sigma > 0
    n = size(x,1);
    y = y + sigma*randn([n,1]);
    if nargout > 1
        s = sigma*ones(n,1);
    end
end
