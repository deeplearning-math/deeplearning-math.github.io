function out=dctmod(in)
	in.coeffs = reshape(in.coeffs,[size(in.mask) numel(in.coeffs)/prod(size(in.mask))]);

	inmask = in.mask;
	out=in;
	
	mask=(sum(inmask==1));
        out.coeffs=zeros(size(out.coeffs));

	for sup=unique(mask)
		if sup>1
			I=find(mask==sup);
			II=find(inmask(:,I(1))==1);
			II=II(end:-1:1);
			tt=zeros(size(inmask));
			tt(II,I) = 1;
			slice=reshape(in.coeffs(II,I,:),[length(II) length(I)*size(in.coeffs,3)]);
			out.coeffs(II,I,:)=reshape(dct(slice),[length(II) length(I) size(in.coeffs,3)]);
		end
	end
	
	out.coeffs = reshape(out.coeffs,[size(in.mask,1) numel(out.coeffs)/size(in.mask,1)]);
end

