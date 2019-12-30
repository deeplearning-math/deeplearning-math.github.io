function [fullcoeffs,fullcoeffs_mask,coeffs_orig]=newdctscatt(transf,options)
%for each scatt order m, rearrange coeffs of order m
%in a 2m-dimensional array, indexed either by (j1,theta1,j2,theta2,...,jm,thetam),
%either by (j1,theta1,j2-j1,theta2-theta1,...,jm-j_{m-1},thetam-theta_{m-1})

	L = getoptions(options, 'L', 6);
	J = getoptions(options, 'J', 3);
	S=size(transf);
	skip_first_order=getoptions(options,'skip_first_order',2);

	if ~isfield(options,'scatdims')
		[N,M]=size(transf{1}{1}.signal);
		options.scatdims=[1 N 1 M];
	else
		N=options.scatdims(3)-options.scatdims(1)+1;
		M=options.scatdims(4)-options.scatdims(2)+1;
	end

	onedim=0;
	if M==1 || N==1
		onedim=1;
		L=1;
	end

	differential_coding_or=getoptions(options,'differential_dct_or',1);
	differential_coding_sc=getoptions(options,'differential_dct_sc',1);
	transf_mode=getoptions(options,'dct_transf_mode',0);
	flip_transf_order=getoptions(options,'flip_order',1);

	cutoff_freq=getoptions(options,'dct_cutoff',0);
	if cutoff_freq
		cutoff=getoptions(options,'dct_freqs',standard_cutoff_table(S(2),3,3));
	else
		cutoff=standard_cutoff_table(S(2),max(L,J)+1,max(L,J)+1);
	end

	numpixels=N*M;
	insize_or=[];
	insize_sc=[];

	tmp=transf{1}{1}.signal(options.scatdims(1):options.scatdims(3),options.scatdims(2):options.scatdims(4));
	fullcoeffs{1}=tmp(:);
	coeffs_orig{1}=tmp(:);

	for m=2:S(2)
		insize_or=[L insize_or];
		insize_sc=[J insize_sc];
		if onedim
			insize=[numpixels insize_sc];
		else
			insize=[numpixels insize_or insize_sc];
		end
		fullcoeffs{m}=zeros(insize);
		fullcoeffs_mask{m}=logical(zeros(insize));
		%fill with coeffs
		SS=size(transf{m});
		for ss=1:SS(2)
			%find the raster index
			code_or = transf{m}{ss}.meta.orientation;
			code_sc = transf{m}{ss}.meta.scale;
			if differential_coding_or
				code_ori = myind2sub(transf{m}{ss}.meta.orientation,L,m-1);
				code_shifted=zeros(1,m-1);
				if m>2
					code_shifted(1:m-2) = code_ori(2:end);
				end
				code_combined = mod(code_ori - code_shifted,L);
				code_or = mysub2ind(code_combined,L,m-1);
			end
			if differential_coding_sc
				code_ori = myind2sub(transf{m}{ss}.meta.scale,J,m-1);
				code_shifted=zeros(1,m-1);
				if m>2
					code_shifted(1:m-2) = code_ori(2:end);
				end
				code_combined = mod(code_ori - code_shifted,J);
				code_sc = mysub2ind(code_combined,J,m-1);
				%code_sc = transf{m}{ss}.meta.scale - floor(transf{m}{ss}.meta.scale/J);
			end
	
			if onedim
				raster = numpixels*code_sc;
			else
				raster = numpixels*code_or + numpixels*L^(m-1)*code_sc;
			end

			tmp=transf{m}{ss}.signal(options.scatdims(1):options.scatdims(3),options.scatdims(2):options.scatdims(4));
			fullcoeffs{m}(raster+1:raster+numpixels)=tmp(:);
			fullcoeffs_mask{m}(raster+1:raster+numpixels)=1;
		end
		%transform along each direction
		coeffs_orig{m}=fullcoeffs{m};

		switch transf_mode
		case 0
			transf_array=1:2*(m-1);%transform along both scale and orientation
		case 1
			transf_array=1:1*(m-1);%transform along orientation
		case 2
			transf_array=m:2*(m-1);%transform along scale
		end
		if onedim
			%transf_array=1:m-1;
			transf_array=m-1:-1:1;
		end
		if flip_transf_order
			transf_array=fliplr(transf_array);
		end
		%if m==3
		%	transf_array=4;
		%end

		numofdims=length(size(fullcoeffs{m}));
		for dim=transf_array
			tempo=shiftdim(fullcoeffs{m},dim);
			tempo_mask=shiftdim(fullcoeffs_mask{m},dim);
			tempsize=size(tempo);
			tempo=reshape(tempo,size(tempo,1),numel(tempo)/size(tempo,1));
			tempo_mask=reshape(tempo_mask,size(tempo,1),numel(tempo)/size(tempo,1));

			[tempo,tempo_mask]=dctmod(tempo,tempo_mask,cutoff{m}(dim));
			% tempsize(1)=size(tempo,1);
			% if cutoff_freq
			%   tempsize(1)=min(tempsize(1),cutoff{m}(dim));
			%   tempo=tempo(1:tempsize(1),:);
			%   tempo_mask=tempo_mask(1:tempsize(1),:);
			% end
			% tempsize(1)

			tempo=reshape(tempo,tempsize);
			tempo_mask=reshape(tempo_mask,tempsize);
			fullcoeffs{m}=shiftdim(tempo,numofdims-dim);
			fullcoeffs_mask{m}=shiftdim(tempo_mask,numofdims-dim);
		end
	end
end


function out=standard_cutoff_table(M,orcut_in,sccut_in)
	huge=1024;

	for m=2:M
		orcut=(m==2)*huge + (m>2)*orcut_in;
		sccut=(m==2)*huge + (m>2)*sccut_in;
		orvect=orcut*ones(1,m-1);
		orvect(m-1)=huge;
		scvect=sccut*ones(1,m-1);
		scvect(m-1)=huge;
		out{m}=[orvect scvect];
	end
end
