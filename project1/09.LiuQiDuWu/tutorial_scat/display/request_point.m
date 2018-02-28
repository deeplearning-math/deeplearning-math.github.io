function out=request_points(px,py,meta)

C=length(meta.order);
out=zeros(size(px));
for c=1:C
	isin=(meta.rectangle(c,1) < px).*(px <= meta.rectangle(c,2)).*(meta.rectangle(c,3) < py).*(py <=meta.rectangle(c,4)); 
	out=out+isin*meta.norm_ratio(c);
end
