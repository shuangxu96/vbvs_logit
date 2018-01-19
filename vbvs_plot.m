function vbvs_plot(model,type,lw,fs)
if ~exist('lw','var'); lw = 3;end
if ~exist('fs','var'); fs = [];end
if ~exist('type','var'); type = 'trace'; end
if ~isfield('fs','title'); fs.title = 20; end
if ~isfield('fs','xlabel'); fs.xlabel = 20; end
if ~isfield('fs','ylabel'); fs.ylabel = 20; end

if isequal(type,'trace')
    vbvs_plot_trace(model,lw,fs)
end
if isequal(type,'bic')
    vbvs_plot_bic(model,lw,fs)
end

end

function vbvs_plot_trace(model,lw,fs)
xdata = model.rho;
ydata = model.MU';
dashline = model.rhomin;
plot(xdata,ydata,'LineWidth', lw)
xlim([min(xdata),max(xdata)])
hold on
plot([dashline,dashline], get(gca, 'YLim'), '--', 'LineWidth', 1)
hold off
title('Coefficient Traces','FontSize',fs.title),  
ylabel('Coefficients','FontSize',fs.ylabel)
xlabel('$\rho$', 'interpreter', 'latex','FontSize',fs.xlabel)
end

function vbvs_plot_bic(model,lw,fs)
xdata = model.rho;
ydata = model.BIC;
dashline = model.rhomin;

plot(xdata,ydata,'LineWidth', lw)
hold on
plot([dashline,dashline], get(gca, 'YLim'), '--', 'LineWidth', 1)
hold off
ylabel('BIC','FontSize',fs.ylabel)
xlabel('$\rho$', 'interpreter', 'latex','FontSize',fs.xlabel)

end