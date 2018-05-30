% C. Zhang, S. Xu and J. Zhang. A Novel Variational Bayesian Method for
% Variable Selection in Logistic Regression Models. 2018

function model = vbvs_logit(X, y, nrho, rhomin, rhomax, a0, b0, maxiter, tol)
% A Variational Bayesian Variable Selection Method for Logistic Regression
% Models:
% p(y = 1 | x, \beta) = 1 / (1 + exp(- beta' * Gamma * x))
% with prior
% p(beta | diag[alpha]) = Gauss(\beta | 0, diag[alpha]^-1),
% p(alpha_j) = Gam(alpha_j | a0, b0),
% p(gamma_j) = Ber(gamma_j | rho).
% The hyper-parameter rho controls the model sparity. Three criteria are
% available to tune it. But we suggest to use BIC.
%
% *input*
% X      : data matrix with shape n*p, where n and p are the number of
%          samples and features, respectively.
% y      : response vector with shape n*1, the element of which must be 1 or
%          -1.
% nrho   : the number of candidates of rho. (optional; defualt: 100)
% rhomin : the lower bound of rho.(optional; defualt: -10)
% rhomax : the upper bound of rho.(optional; defualt: 3)
% a0     : the hyper-parameter in p(alpha) with shape p*1.(optional; defualt: 1e-2*ones(p,1))
% b0     : the hyper-parameter in p(alpha) with shape p*1.(optional; defualt: 1e-4*ones(p,1))
% maxiter: the maximum value of iteration.(optional; defualt: 100)
% tol    : the tolerance for convergence.(optional; defualt: 1e-3)
%
% *output*
% model  : the inferred model (struct).
% model.rho: the candidates of rho.
% model.allmodels: the model details.
% model.allcoef: the regression coefficients with shape p*nrho. 
% model.VBIC: the variational BIC of each rho.
% model.ELBO: the ELBO of each rho.
% model.BIC: the BIC of each rho.
% model.BICmodel: the optimal model selected by BIC.
% model.mu: the regression coefficient of the BIC model.
%
% Copyright (c) 2018 Shuang Xu
% All rights reserved.
%--------------------------------------------------------------------------
% check response
uy = unique(y);
if length(uy)==1
    error('There is just one class!')
end
if ~isempty(setdiff(uy,[-1,1]))
    error('Y is illegal. Only -1 or 1 is permmited. ')
end

% check arguments
[~, p] = size(X); model.X = X; model.y = y;
if ~exist('nrho','var'); nrho = 100;end
if ~exist('rhomin','var'); rhomin = -10;end
if ~exist('rhomax','var'); rhomax = 3;end
if ~exist('a0','var'); a0 = 1e-2*ones(p,1);end
if ~exist('b0','var'); b0 = 1e-4*ones(p,1);end
if ~exist('maxiter','var'); maxiter = 100;end
if ~exist('tol','var'); tol = 1e-3;end

% choose optimal rho
rho = flip(logsig(linspace(rhomin, rhomax, nrho)));
BIC = zeros(nrho,1);
[VBIC, ELBO] = deal(BIC, BIC);
allmodels = cell(nrho,1);
allcoef = zeros(p+1, nrho);
for i = 1:nrho
    fit = vbvs_logit_fit(X, y, rho(i), a0, b0, maxiter, tol);
    allcoef(:,i) = fit.coef;
    [VBIC(i), ELBO(i), BIC(i)] = deal(fit.VBIC,fit.ELBO(end),fit.BIC);
    allmodels{i} = fit;
end

% output
model.rho = rho';
model.allmodels = allmodels;
model.allcoef = allcoef;
model.VBIC = VBIC;
model.ELBO = ELBO;
model.BIC = BIC;
model.BICmodel = vbvs_logit_opt(model,'BIC');
model.mu = model.BICmodel.coef;