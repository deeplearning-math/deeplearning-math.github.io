function plot_topo_scatter(in,in_order,with_log)
	if nargin < 3
		with_log = 0;
	end
	
	color(1) = 'c';
	color(2) = 'r';
	color(3) = 'g';
	color(4) = 'b';
	color(5) = 'm';
	color(6) = 'y';
	color(7:99) = 'k';
	
	if with_log
		in = log10(abs(in));
	end
	
	clf;
	hold on;
	for k = 1:length(in)-1
		if k > 1
			plot([k k],[(in(k-1)+in(k))/2 in(k)],[color(in_order(k)+1)]);
		end
		plot([k k+1],[in(k) in(k)],[color(in_order(k)+1)]);
		if k > 1
			plot([k+1 k+1],[in(k) (in(k)+in(k+1))/2],[color(in_order(k)+1)]);
		end
	end
	hold off;
end