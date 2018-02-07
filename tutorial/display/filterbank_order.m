function [out,out_orders,out_scales] = filterbank_order(in)	
	in_scales = cell(size(in));
	
	for m = 1:length(in)
		in_scales{m} = zeros(size(in{m}));
		for k = 1:length(in{m})
			in_scales{m}(k) = in{m}{k}.meta.scale;
		end
	end
	
	[out,out_orders,out_scales] = filterbank_order_helper(in,in_scales,1,0);
end

function [out,out_orders,out_scales] = filterbank_order_helper(in,in_scales,m0,j0)
	nscales = length(in{2});

	if m0 == 1
		out = in{1}{1}.signal;
		out_orders = 0;
		out_scales = -1;
	else
		out = [];
		out_orders = [];
		out_scales = [];
		ind = find(in_scales{m0}==j0);
		if ~isempty(ind)
			out = in{m0}{ind}.signal;
			out_orders = m0-1;
			out_scales = j0;
		end
	end

	if ~isempty(out) && m0 < length(in)
		outb = [];
		outb_orders = [];
		outb_scales = [];
		for j1 = 1:nscales
			if ~isempty(find(in_scales{m0+1}==j0*nscales+j1-1))
				[outp,outp_orders,outp_scales] = filterbank_order_helper(in,in_scales,m0+1,j0*nscales+j1-1);
				outb = [outp outb];
				outb_orders = [outp_orders outb_orders];
				outb_scales = [outp_scales outb_scales];
			end
		end
		
		out = [out outb];
		out_orders = [out_orders outb_orders];
		out_scales = [out_scales outb_scales];
	end
end