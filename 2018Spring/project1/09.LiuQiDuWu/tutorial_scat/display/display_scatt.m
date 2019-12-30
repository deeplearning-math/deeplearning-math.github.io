function display_scatt(in,meta,options)

%this function displays scattering coefficients 
%up to order 2

SI=length(size(in));

first_mask=find(meta.order==2);
J=max(meta.scale(first_mask))+1
L=max(meta.orientation(first_mask))+1

marge=0.1;
unit=100;
corr_fact=1.2;
border=1;
lpos=1/L;
jpos=1/J;
lsize=lpos*(1-marge);
jsize=jpos*(1-marge);
loffs=lpos*marge*.5;
joffs=jpos*marge*.5;



%order 0
ind=find(meta.order==1);
	figure(1);
	title('order 0');

if SI==3
	imagesc(in(:,:,ind));
	colormap gray
else
	plot(in(:,ind));
end

%order 1
ind=find(meta.order==2);
figure(2);
title('order 1');
disp_b=-3000;

	for i=ind
		j=meta.scale(i);
		l=meta.orientation(i);
		subplot('position',[loffs+l*lpos joffs+(J-j-1)*jpos loffs+lsize joffs+jsize]); 
		if SI==3
			imagesc(bbox(64+disp_b*in(:,:,i),border));
			axis off
			colormap gray
		else
			plot(in(:,i));
		end
	end


%order 2
ind=find(meta.order==3);
figure(3);
title('order 2');
disp_b=-5000;

lpos=1/L^2;
jpos=1/J^2;
lsize=lpos*(1-marge);
jsize=jpos*(1-marge);
loffs=lpos*marge*.5;
joffs=jpos*marge*.5;



for i=ind

		j1=mod(meta.scale(i),J);
		l1=mod(meta.orientation(i),L);
		j2=mod(floor(meta.scale(i)/J),J);
		l2=mod(floor(meta.orientation(i)/L),L);

		subplot('position',[loffs+l1*lpos+l2*L*lpos joffs+(J-j1-1)*jpos+J*(J-j2-1)*jpos loffs+lsize joffs+jsize]); 
		if SI==3
			imagesc(bbox(64+disp_b*in(:,:,i),border));
			axis off
			colormap gray
		else
			plot(in(:,i));
		end


end



