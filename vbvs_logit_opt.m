% C. Zhang, S. Xu and J. Zhang. A Novel Variational Bayesian Method for
% Variable Selection in Logistic Regression Models. 2018

function object = vbvs_logit_opt(model, s)
% Select the optimal rho.
% *input*
% model: the output of vbvs_logit.
% s    : a criterion, either 'BIC', 'VBIC' or 'ELBO'. (optional; default: 'BIC')
% *output*
% object: the details of optimal model.

if nargin == 1
    s = 'BIC';
end
if strcmpi(s, 'BIC')
    value = model.BIC;
elseif strcmpi(s, 'VBIC')
    value = model.VBIC;
elseif strcmpi(s, 'ELBO')
    value = -model.ELBO;
else
    error('s is not in {BIC,VBIC,ELBO}')
end

[~,ind] = min(value);
object.optindex = ind;
object.optrho = model.rho(ind);
object.optmodel = model.allmodels{ind};
object.coef = object.optmodel.coef;