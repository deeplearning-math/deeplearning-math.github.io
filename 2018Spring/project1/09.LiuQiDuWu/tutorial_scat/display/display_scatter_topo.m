function [out,out_order,out_scale,in,in_measure,in_order,in_scale,Ein,Edirac] = display_scatter_topo(s,options,K,with_log)
	% Displays the topological plot of the scattering vector of 's', computed using the parameters in 'options'
	% using 'K' points. If 'with_log' is set to 1, the logarithm of the plot is shown.
	
	if nargin < 3
		M = 2*length(s);
	end
	
	if nargin < 4
		with_log = 0;
	end
	
	t = newscatt(s(:),options);
	[t,t_order,in_scale] = filterbank_order(t);
	tdirac = newscatt([1; zeros(length(s)-1,1)],options);
	[tdirac,tdirac_order] = filterbank_order(tdirac);
	Edirac = sum(abs(tdirac).^2,1);
	Ein = sum(abs(t).^2,1);
	in_order = t_order;
	fprintf('Total dirac measure: %f\n',sum(Edirac));
	[out,out_order,out_scale,in,in_measure] = apply_topo(sqrt(Ein),t_order,in_scale,Edirac,K);
	
	plot_topo_scatter(out,out_order,with_log);
end