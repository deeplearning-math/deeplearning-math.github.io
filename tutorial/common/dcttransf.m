function [out,inmask]=dcttransf(in,dim,m,J,L);
%dim goes to 0 to 2*m-1. 
%it indicates the coordinate along which we will 
%transform. it will be used only to record the path
%in the meta information

	out=in;
	[K,L1]=size(in.coeffs);
	[K2,L2]=size(in.mask);
	npixels=L1/L2;

	coeffs=reshape(in.coeffs,K,L2,npixels);
	hscslice=zeros(size(in.code_sc_hsc));
	
        inweight=in.weight;
	inmask=in.mask;

	mask=(sum(inmask==1));
	outweight=inweight;

	for sup=unique(mask)
		if sup>1
			I=find(mask==sup);
			II=find(inmask(:,I(1))==1);
			slice=coeffs(II,I,:);
                        sliceb=reshape(slice,size(slice,1),numel(slice)/size(slice,1));
                        cout=dct(sliceb);
			coeffs(II,I,:)=reshape(cout,size(slice));
                        hscslice(II,I)=([0:size(cout,1)-1]')*ones(1,length(I));
		end
	end

	out.weight=outweight(:,1:L2);
	out.coeffs=reshape(coeffs,size(out.coeffs));

	%encode dct transform
	if dim < m
		%transform along orientations
		out.code_or_hsc = out.code_or_hsc + L^dim*hscslice(:,1:L2);
	else
		%transform along scales
		out.code_sc_hsc = out.code_sc_hsc + J^(dim-m)*hscslice(:,1:L2);
	end

end
