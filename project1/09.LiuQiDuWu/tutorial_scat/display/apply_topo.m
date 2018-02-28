function [out,out_order,out_scale,in,in_measure] = apply_topo(in,in_order,in_scale,Edirac,N)
	if nargin < 4
		N = length(in(:));
	end
	
	in1 = zeros(size(in));
	in_measure = zeros(1,size(in,2));
	
	out = ones(size(in,1),N)*NaN;
	out_order = zeros(1,N);
	out_scale = zeros(1,N);
	
	p = 0;
	for k = 1:size(in,2)
		%out(1+floor(p):1+floor(p+Edirac(k)*N)) = sqrt(sum(abs(in(:,k)).^2)/Edirac(k));
		%out_order(1+floor(p):1+floor(p+Edirac(k)*N)) = in_orders(k);
		%p = p+Edirac(k)*N;
		
		%in1(k) = sqrt(sum(abs(in(:,k)).^2))/(1e-12+sqrt(Edirac(k)));
		in1(:,k) = in(:,k)/(1e-12+sqrt(Edirac(k)));
		%out_order(k) = in_order(k);
		in_measure(k) = p;
		p = p+Edirac(k)*N;
	end
	in = in1;
	in_measure(end+1) = p;
	
	for k = 1:N
		start_ind = k-1<=in_measure(1:end-1)&in_measure(1:end-1)<=k;
		stop_ind = k-1<=in_measure(2:end)&in_measure(2:end)<=k;
		
		ind = start_ind|stop_ind;
		ind = find(ind);
		
		if isempty(ind)
			ind = find(in_measure(1:end-1)<=k-1&in_measure(2:end)>=k);
		end
		
		if isempty(ind)
			continue;
		end
		
		interval_magnitude = zeros(size(in,1),length(ind));
		interval_measure = zeros(1,length(ind));
		interval_order = zeros(1,length(ind));
		interval_scale = zeros(1,length(ind));
		
		interval_order_measure = zeros(1,100);
		for r = 1:length(ind)
			e1 = max(k-1,in_measure(ind(r)));
			e2 = min(k,in_measure(ind(r)+1));
			interval_magnitude(:,r) = in(:,ind(r));
			interval_measure(r) = e2-e1;
			interval_order(r) = in_order(ind(r));
			interval_scale(r) = in_scale(ind(r));
			interval_order_measure(interval_order(r)+1) = interval_order_measure(interval_order(r)+1)+(e2-e1);
		end
		
		%out(:,k) = sum(bsxfun(@times,interval_magnitude,interval_measure),2)/sum(interval_measure);
		%[temp,order] = max(interval_order_measure);
		%out_order(k) = order-1;
		[temp,ind] = max(interval_measure);
		out(:,k) = interval_magnitude(:,ind);
		out_order(k) = interval_order(ind);
		out_scale(k) = interval_scale(ind);
	end
end