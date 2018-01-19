function model = vbvs_logit(X, y, opt)
% [model] = vbvs_logit_fit(X, y, opt)
%
% Create sparse logit regression model by fitting to data,
% $$p(y = 1 | x, \beta) = 1 / (1 + exp(- \beta' * \Gamma * x))$$
% with prior
% p(beta | a) = p(\beta | 0, a^-1 I),
% p(a_j) = Gam(a_j | a0, b0),
% p(gamma_j | rho_j) = Ber(gamma_j | rho_j),
% p(rho_j) = Beta(rho_j | f0, g0).
%
% *Inputs* :
% X - observation matrix, [n x p].
% y - output vector, containing either 1 or -1, [n x 1].
% opt.nrho - the number of rho, (Default: 100), [1 x 1].
% opt.rhomin - the minimum of log(rho/(1-rho)), (Default: -10), [1 x 1].
% opt.rhomax - the maximum of log(rho/(1-rho)), (Default: 3), [1 x 1].
% opt.theta - initial theta, [p x 1].
% opt.a0 - hyper-parameter, (Default: 1e-2), [p x 1];
% opt.b0 - hyper-parameter, (Default: 1e-4), [p x 1];
% opt.maxiter - maximum number of iteration, (Default: 500), [1 x 1];
% opt.tol - tolerance of error, (Default: 1e-4), [1 x 1];
% opt.intercept - intercerpt indicator, either true or false, (Default:
% 'true'), [1 x 1]. If intercept == true/false, model is with/without
% intercept iterm.
%
% *Output* :
% model - sparse logistic regression model.
%
% Copyright (c) 2017, Shuang Xu
% All rights reserved.
% See the file LICENSE for licensing information.


% check response
uy = unique(y);
if length(uy)==1
    error('There is just one class!')
end
if ~isempty(setdiff(uy,[-1,1]))
    error('Y is illegal. Only -1 or 1 is permmited. ')
end
% pre-process parameter
if ~exist('opt','var'); opt = struct();end
if (~isfield(opt,'nrho')); nrho = 100; else; nrho = opt.nrho; end
if (~isfield(opt,'rhomin')); rhomin = -10; else; rhomin = opt.rhomin; end
if (~isfield(opt,'rhomax')); rhomax = 3; else; rhomax = opt.rhomax; end
if (~isfield(opt,'intercept')); intercept = true; else; intercept = opt.intercept; end
[n, p] = size(X); model.X = X; model.y = y;
if intercept; X = [ones(n,1),X];end
rho = logsig(linspace(rhomin, rhomax, nrho));
BIC = zeros(nrho,1);
Model = cell(nrho,1);
MU = zeros(p+intercept,nrho);

% fit models
for (i = 1:nrho)
    [fit] = vbvs_logit_fit_mex_pre(X, y, rho(i), opt);
    indicator = fit.theta;
    indicator(indicator<0.5) = 0; indicator(indicator>=0.5) = 1; 
    indicator = logical(indicator);
    p1 = sum(indicator~=0);
    mu = zeros(size(indicator));
    mu(indicator) = fit.mu(indicator);
    MU(:,i) = mu;
    BIC(i) = 2 * sum(log(1+exp( - y .* (X * mu)))) + p1 * log(n);
    fit.indicator = indicator;
    Model{i} = fit;
end
[~,ind] = min(BIC);
mopt = Model{ind};

% select optimal model
indicator = mopt.indicator;
mu = zeros(p+intercept,1);
mu(indicator) = mopt.mu(indicator);
Sigma = mopt.Sigma;
invSigma = inv(Sigma);

% output
model.modelOPT = mopt;
model.rho = rho';
model.rhomin = rho(ind);
model.rhominIndex = ind;
model.Importance = mopt.indicator;
model.intercept = intercept;
model.Sigma = Sigma;
model.invSigma = invSigma;
model.mu = mu;
model.MU = MU;
model.BIC = BIC;
model.allmodel = Model;
end

function [model] = vbvs_logit_fit_mex_pre(X, y, rho, opt)
[~,p] = size(X);
if ~exist('opt','var'); opt = struct(); end
if (~isfield(opt,'theta')); theta = ones(p,1); else; theta = opt.theta; end
if (~isfield(opt,'a0')); a0 = 1e-2*ones(p,1); else; a0 = opt.a0; end
if (~isfield(opt,'b0')); b0 = 1e-4*ones(p,1); else; b0 = opt.b0; end
if (~isfield(opt,'maxiter')); maxiter = 500; else; maxiter = opt.maxiter; end
if (~isfield(opt,'tol')); tol = 1e-4; else; tol = opt.tol; end

model = vbvs_logit_fit_mex(X, y, rho, theta, a0, b0, maxiter, tol);
model.HistoryELBO = model.HistoryELBO(1:(model.Step-1));
end