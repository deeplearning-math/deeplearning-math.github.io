function [out_first,out_scales,out_orientations]=display_2d_split(meta,average_scatt)
%we assume here that scatt and average_scatt have the same parameters

%this function produces three images;
%(i) first order coeffs, in (j,theta)
%(ii) second order coeffs, in a mosaic of LxL small plots, as a function of (j1,j2)
%(iii) second order coeffs, in a mosaic of JxJ small plots, as a function of (theta1,theta2)

%it can easily be adapted to produce the marginalized versions (average images)

first_mask=find(meta.order==2);
second_mask=find(meta.order==3);
J=max(meta.scale(first_mask))+1;
L=max(meta.orientation(first_mask))+1;
maxsize=2048;

meta=add_average_process(meta,average_scatt);
meta=effective_energy(meta);

%first order display
meta1=compute_rectangles_1(meta);

heights=meta1.rectangle(first_mask,2)-meta1.rectangle(first_mask,1);
widths=meta1.rectangle(first_mask,4)-meta1.rectangle(first_mask,3);
fact_h=min(ceil(4/min(heights)),maxsize);
fact_w=min(ceil(4/min(widths)),maxsize);
out_first=zeros(fact_h,fact_w);
for l=first_mask
	inth=min(fact_h,[1+floor(fact_h*meta1.rectangle(l,1)):floor(fact_h*meta1.rectangle(l,2))]);
	intw=min(fact_w,[1+floor(fact_w*meta1.rectangle(l,3)):floor(fact_w*meta1.rectangle(l,4))]);
	out_first(inth,intw)=sqrt(meta.sq_scatt_ave(l)/meta.average(l));
end

%second order display
meta2o=compute_rectangles_scale(meta);
heights=meta2o.rectangle(first_mask,2)-meta2o.rectangle(first_mask,1);
widths=meta2o.rectangle(first_mask,4)-meta2o.rectangle(first_mask,3);
fact_h=min(ceil(4/min(heights)),maxsize);
fact_w=min(ceil(4/min(widths)),maxsize);
out_scales=zeros(fact_h,fact_w);
for l=second_mask
	inth=min(fact_h,[1+floor(fact_h*meta2o.rectangle(l,1)):floor(fact_h*meta2o.rectangle(l,2))]);
	intw=min(fact_w,[1+floor(fact_w*meta2o.rectangle(l,3)):floor(fact_w*meta2o.rectangle(l,4))]);
	out_scales(inth,intw)=sqrt(meta.sq_scatt_ave(l)/meta.average(l));
end

meta2s=compute_rectangles_orientation(meta);
heights=meta2s.rectangle(first_mask,2)-meta2s.rectangle(first_mask,1);
widths=meta2s.rectangle(first_mask,4)-meta2s.rectangle(first_mask,3);
fact_h=min(ceil(4/min(heights)),maxsize);
fact_w=min(ceil(4/min(widths)),maxsize);
out_orientations=zeros(fact_h,fact_w);
for l=second_mask
	inth=min(fact_h,[1+floor(fact_h*meta2s.rectangle(l,1)):floor(fact_h*meta2s.rectangle(l,2))]);
	intw=min(fact_w,[1+floor(fact_w*meta2s.rectangle(l,3)):floor(fact_w*meta2s.rectangle(l,4))]);
	out_orientations(inth,intw)=sqrt(meta.sq_scatt_ave(l)/meta.average(l));
end


end

function meta=compute_rectangles_1(meta)
%this function displays normalized scattering first order coeffs

R=length(meta.order);
maxorder=max(meta.order);
first_mask=find(meta.order==2);
J=max(meta.scale(first_mask))+1;
L=max(meta.orientation(first_mask))+1;

meta.rectangle(1,:)=[0 1 0 1];

for o=1:2%maxorder-1
	slice=find(meta.order==o);
	for s=slice
		%find children
		if o==1
			children=find(meta.order==o+1);
		else
			children=find((floor(meta.scale/J)==meta.scale(s))&(floor(meta.orientation/L)==meta.orientation(s))&(meta.order==o+1));
		end
		if ~isempty(children)
			[newrectangles,outrect]=split_rectangle(meta.rectangle(s,:),meta.scale(children),meta.orientation(children),...
				meta.effnorms(children),J,L);
			for c=1:length(children)
				meta.rectangle(children(c),:)=newrectangles(c,:);
			end
			meta.rectangle(s,:)=outrect;
		end
	end
end

end


function meta=compute_rectangles_orientation(meta)
%this function displays the normalized scattering in 2d in a single plot

R=length(meta.order);
maxorder=max(meta.order);
first_mask=find(meta.order==2);
J=max(meta.scale(first_mask))+1;
L=max(meta.orientation(first_mask))+1;

init_rectangle=[0 1 0 1];

for l1=1:L
	for l2=1:L
		pack=find((meta.order==3)&(meta.orientation==l1-1+L*(l2-1)));
		ener(l1+L*(l2-1))=sum(meta.effnorms(pack));
		orient1(l1+L*(l2-1))=l1-1;
		orient2(l1+L*(l2-1))=l2-1;
	end
end

[first_rectangles]=split_rectangle(init_rectangle,orient1,orient2,ener,L,L);
%shrink the rectangles a little bit in order to see a clear separation in the display
rho=0.8;
first_rectangles(:,2)=first_rectangles(:,1)+rho*(first_rectangles(:,2)-first_rectangles(:,1));
first_rectangles(:,4)=first_rectangles(:,3)+rho*(first_rectangles(:,4)-first_rectangles(:,3));


for l1=1:L
	for l2=1:L
		pack=find((meta.order==3)&(meta.orientation==l1-1+L*(l2-1)));
		%rectangles=split_rectangle(first_rectangles(l1+L*(l2-1),:),mod(meta.scale(pack),J),mod(floor(meta.scale(pack)/J),J),...
		rectangles=split_rectangle(first_rectangles(l1+L*(l2-1),:),...
		mod(meta.scale(pack)-floor(meta.scale(pack)/J),J),mod(floor(meta.scale(pack)/J),J),...
			meta.effnorms(pack),J,J);
		for c=1:length(pack)
			meta.rectangle(pack(c),:)=rectangles(c,:);
		end
	end
end

end


function meta=compute_rectangles_scale(meta)
%this function displays the normalized scattering in 2d in a single plot

R=length(meta.order);
maxorder=max(meta.order);
first_mask=find(meta.order==2);
J=max(meta.scale(first_mask))+1;
L=max(meta.orientation(first_mask))+1;

init_rectangle=[0 1 0 1];

for j1=1:J
	for j2=1:J
		pack=find((meta.order==3)&(meta.scale==j1-1+J*(j2-1)));
		if ~isempty(pack)
			ener(j1+J*(j2-1))=sum(meta.effnorms(pack));
			orient1(j1+J*(j2-1))=j1-1;
			orient2(j1+J*(j2-1))=j2-1;
		end
	end
end

[first_rectangles]=split_rectangle(init_rectangle,orient1,orient2,ener,J,J);
%shrink the rectangles a little bit in order to see a clear separation in the display
rho=0.8;
first_rectangles(:,2)=first_rectangles(:,1)+rho*(first_rectangles(:,2)-first_rectangles(:,1));
first_rectangles(:,4)=first_rectangles(:,3)+rho*(first_rectangles(:,4)-first_rectangles(:,3));

for j1=1:J
	for j2=1:J
		pack=find((meta.order==3)&(meta.scale==j1-1+J*(j2-1)));
		if ~isempty(pack)
			%rectangles=split_rectangle(first_rectangles(j1+J*(j2-1),:),mod(meta.orientation(pack),L),...
			rectangles=split_rectangle(first_rectangles(j1+J*(j2-1),:),mod(meta.orientation(pack)-floor(meta.orientation(pack)/L),L),...
			mod(floor(meta.orientation(pack)/L),L),...
			meta.effnorms(pack),L,L);
		for c=1:length(pack)
			meta.rectangle(pack(c),:)=rectangles(c,:);
		end
		end
	end
end

end


function [out,outlowp]=split_rectangle(inrectangle, scales, orientations, dirac_norms,J,L)

%first step: we marginalize orientations in order to split scale axis:
C=length(dirac_norms);
totener=sum(dirac_norms);

totalheight=inrectangle(2)-inrectangle(1);
totalwidth=inrectangle(end)-inrectangle(end-1);

outlowp=inrectangle;
rasterwidth=inrectangle(3);

for j=J-1:-1:0
	pack=find(mod(scales,J)==j);
	if ~isempty(pack)
		width=sum(dirac_norms(pack).^1);
		rasterheight=inrectangle(1);
		for l=0:L-1
			ind=find(mod(orientations(pack),L)==l);
			if ~isempty(ind)
			out(pack(ind),1)=rasterheight;
			out(pack(ind),2)=rasterheight+totalheight*dirac_norms(pack(ind)).^1/width;
			out(pack(ind),3)=rasterwidth;
			out(pack(ind),4)=rasterwidth+totalwidth*width/totener;
			rasterheight=out(pack(ind),2);
			end
		end
		rasterwidth=rasterwidth+totalwidth*width/totener;
	end
end

end



function metaout=effective_energy(meta)


R=length(meta.order);
maxorder=max(meta.order);
first_mask=find(meta.order==2);
J=max(meta.scale(first_mask))+1;
L=max(meta.orientation(first_mask))+1;

last_mask=find(meta.order==maxorder);
metaout=meta;
metaout.effnorms(last_mask)=meta.average(last_mask).^1;

for o=maxorder-1:-1:1
	slice=find(meta.order==o);
	for s=slice
		%find children
		if o==1
			children=find(meta.order==o+1);
		else
			children=find((floor(meta.scale/J)==meta.scale(s))&(floor(meta.orientation/L)==meta.orientation(s))&(meta.order==o+1));
		end
		metaout.effnorms(s)=sum(metaout.effnorms(children))+meta.average(s).^1;
	end
end

end


function metaout=add_average_process(meta,average_scatt)

L=length(meta.order);
metaout=meta;
for l=1:L
	metaout.average(l)=average_scatt(l);
end

end


