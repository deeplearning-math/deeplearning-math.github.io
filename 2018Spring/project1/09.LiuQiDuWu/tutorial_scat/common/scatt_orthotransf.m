function [out,outmeta]=scatt_orthotransf(transf,meta,options)
%for each scatt order m, rearrange coeffs of order m
%in a 2m-dimensional array, indexed either by (j1,theta1,j2,theta2,...,jm,thetam),
%either by (j1,theta1,j2-j1,theta2-theta1,...,jm-j_{m-1},thetam-theta_{m-1})
	options.null=1;
        outmeta=meta;

	S=size(transf);

	transf_type=getoptions(options,'transf_type','dct');				% Haar transform or DCT?
	differential_coding_or=getoptions(options,'differential_dct_or',0);
	differential_coding_sc=getoptions(options,'differential_dct_sc',1);
	transf_mode=getoptions(options,'dct_transf_mode',0);
	renormalization=getoptions(options,'renormalization',1);

	flip_transf_order=getoptions(options,'transf_order',0);

	if(length(S)==3)		% Bidimensional input, combine into one dimension
		numpixels=S(1)*S(2);
		transf=reshape(transf,S(1)*S(2),S(3));
	elseif(length(S)==2)
		numpixels=S(1);
	else
		error('unsupported input')
	end

	insize_or=[];
	insize_sc=[];

	sec_order=find(meta.order==2);
	M=max(meta.order)-1;
	J=max(meta.scale(sec_order))+1;
	L=max(meta.orientation(sec_order))+1;
	fullcoeffs{1}=transf(:,sec_order);
	maxorder=max(meta.order);

	if ~isfield(meta,'average_process')
		meta.average_process=sqrt(sum(transf.^2,1));
	end

	if ~renormalization
		meta.average_process=ones(size(meta.order));
	end

	first_order=find(meta.order==1);
	out=transf(:,first_order);
	outmeta.order=ones(1,length(first_order));
	outmeta.code_or_pos = 0;
	outmeta.code_or_hsc = 0;
	outmeta.code_sc_pos = 0;
	outmeta.code_sc_hsc = 0;
        outmeta.combined_scale = 0;
        order_offset=0;

	for m=2:maxorder
		insize_or=[L insize_or];
		insize_sc=[J insize_sc];
		insize=[insize_or insize_sc];
		
		% prepare array to transform
		full.coeffs=zeros([numpixels insize]);
		full.weight=zeros(insize);
		full.mask=zeros(insize);
		full.code_or_pos=zeros(insize);
		full.code_or_hsc=zeros(insize);
		full.code_sc_pos=zeros(insize);
		full.code_sc_hsc=zeros(insize);

		% fill with coeffs in proper order
		plantilla=find(meta.order==m);
		for ss=plantilla
			%find the raster index
			code_or = meta.orientation(ss);
			code_sc = meta.scale(ss);
			if differential_coding_or
				code_ori = myind2sub(code_or,L,m-1);
				code_shifted=zeros(1,m-1);
				if m>2
					code_shifted(1:m-2) = code_ori(2:end);
				end
				code_combined = mod(code_ori - code_shifted,L);
				code_or = mysub2ind(code_combined,L,m-1);
			end
			if differential_coding_sc
				code_ori = myind2sub(code_sc,J,m-1);
				code_shifted=zeros(1,m-1);
				if m>2
				code_shifted(1:m-2) = code_ori(2:end);
				end
				code_combined = mod(code_ori - code_shifted,J);
				code_sc = mysub2ind(code_combined,J,m-1);
			end
			raster = code_or + L^(m-1)*code_sc;
			tmp=transf(:,ss);
			full.coeffs(numpixels*raster+1:numpixels*(raster+1))=tmp(:);
			full.mask(raster+1)=1;
			full.weight(raster+1)=meta.average_process(ss);
			full.code_or_pos(raster+1)=code_or;
			full.code_sc_pos(raster+1)=code_sc;
		end
	
		%transform along each plane
		switch transf_mode
		case 0
			transf_array=0:2*(m-1)-1;%transform along both scale and orientation
		case 1
			transf_array=0:1*(m-1)-1;%transform along orientation
		case 2
			transf_array=m-1:2*(m-1)-1;%transform along scale
		end
		if flip_transf_order
			transf_array=fliplr(transf_array);
		end

		% put 'pixel' dimension last
		full.coeffs = permute(full.coeffs,[2:ndims(full.coeffs) 1]);
		for dim=transf_array
			tempo=structshiftdim(full,dim);
			[tempo,tempsize,tempsizec]=structreshape_fwd(tempo);

			% transform along 1st dimension
			if strcmp(transf_type,'haar')
				[tempo,tempow]=haartransf(tempo,dim,m-1,J,L);
			else
				[tempo,tempow]=dcttransf(tempo,dim,m-1,J,L);
			end

			tempo=structreshape_inv(tempo,tempsize,tempsizec);
			full=structshiftdim(tempo,-dim);
		end
		% recover 'pixel' dimension as first dimension
		full.coeffs = permute(full.coeffs,[ndims(full.coeffs) 1:ndims(full.coeffs)-1]);
	
		chunk=full.coeffs;
		chunk=reshape(chunk,size(chunk,1),numel(chunk)/size(chunk,1));
		valids=find(full.mask(:)==1);
		chunk=chunk(:,valids);
		out=[out chunk];
		outmeta.order=[outmeta.order m*ones(1,size(chunk,2))];
		tmp=full.code_or_pos(:);
		outmeta.code_or_pos = [outmeta.code_or_pos tmp(valids)'];
		tmp=full.code_or_hsc(:);
		outmeta.code_or_hsc = [outmeta.code_or_hsc tmp(valids)'];
		tmp=full.code_sc_pos(:);
		outmeta.code_sc_pos = [outmeta.code_sc_pos tmp(valids)'];
		tmp=full.code_sc_hsc(:);
		outmeta.code_sc_hsc = [outmeta.code_sc_hsc tmp(valids)'];
                supp=find(outmeta.order==m);
                tempo_sc=outmeta.code_sc_hsc(supp)';
                tempo_or=outmeta.code_or_hsc(supp)';
                %comb_reg = L^(m-1)*mysub2ind(fliplr(myind2sub(tempo_sc,J,m-1)),J,m-1)+mysub2ind(fliplr(myind2sub(tempo_or,L,m-1)),L,m-1)+order_offset;
                %order_offset=order_offset+(1*J*L)^(m-1);
                comb_reg = L*sum(myind2sub(tempo_sc,J,m-1),2)+J*sum(myind2sub(tempo_or,L,m-1),2)+order_offset;
                order_offset=order_offset+2*(J*L)*(m-1);
                outmeta.combined_scale = [outmeta.combined_scale comb_reg'];
	end
end

function out=structreshape_inv(in,sizein,sizeinc)
	out.coeffs=reshape(in.coeffs,sizeinc);
	out.mask=reshape(in.mask,sizein);
	out.weight=reshape(in.weight,sizein);
	out.code_or_pos=reshape(in.code_or_pos,sizein);
	out.code_or_hsc=reshape(in.code_or_hsc,sizein);
	out.code_sc_pos=reshape(in.code_sc_pos,sizein);
	out.code_sc_hsc=reshape(in.code_sc_hsc,sizein);
end

function [out,sizeout,sizeoutc]=structreshape_fwd(in)
	sizeoutc=size(in.coeffs);
	sizeout=size(in.mask);

	out.coeffs=reshape(in.coeffs,sizeoutc(1),prod(sizeoutc)/sizeoutc(1));
	out.mask=reshape(in.mask,sizeout(1),prod(sizeout)/sizeout(1));
	out.weight=reshape(in.weight,sizeout(1),prod(sizeout)/sizeout(1));
	out.code_or_pos=reshape(in.code_or_pos,sizeout(1),prod(sizeout)/sizeout(1));
	out.code_or_hsc=reshape(in.code_or_hsc,sizeout(1),prod(sizeout)/sizeout(1));
	out.code_sc_pos=reshape(in.code_sc_pos,sizeout(1),prod(sizeout)/sizeout(1));
	out.code_sc_hsc=reshape(in.code_sc_hsc,sizeout(1),prod(sizeout)/sizeout(1));
end

function out=structshiftdim(in,dim)
	n = ndims(in.coeffs);
	
	order = mod([0:n-2]+dim,n-1)+1;
	
	out.coeffs=permute(in.coeffs,[order n]);
	
	out.weight=permute(in.weight,order);
	out.mask=permute(in.mask,order);
	out.code_or_pos=permute(in.code_or_pos,order);
	out.code_or_hsc=permute(in.code_or_hsc,order);
	out.code_sc_pos=permute(in.code_sc_pos,order);
	out.code_sc_hsc=permute(in.code_sc_hsc,order);
end

function out=myind2sub(code,base,len)
	out=zeros(length(code),len);

	for l=1:len
		out(:,l)=mod(floor(code/(base^(l-1))),base);
	end
end

function out=mysub2ind(code,base,len)
        if ismatrix(code)
          out=zeros(size(code,1),1);
	for l=len:-1:1
		out=base*out+code(:,l);
	end
        else
          out=0;
	for l=len:-1:1
		out=base*out+code(l);
	end
        end
end

