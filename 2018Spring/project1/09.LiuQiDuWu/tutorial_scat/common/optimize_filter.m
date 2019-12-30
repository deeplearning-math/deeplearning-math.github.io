function flt = optimize_filter(f,type)
	if nargin < 2
		type = [];
	end
	
	if iscell(f)
		flt = cell(size(f));
		for k = 1:length(f)
			flt{k} = optimize_filter(f{k},type);
		end

		return;
	end
	
	if isstruct(f) && isfield(f,'psi') && isfield(f,'phi')
		flt = f;
		flt.psi = optimize_filter(f.psi,type);
		flt.phi = optimize_filter(f.phi,type);
		
		return;
	elseif isstruct(f)
		flt = f;
		
		return;
	end
	
	space_threshold = 1e-3;
	fourier_threshold = 1e-6;
	max_blocks = 4;

	sort_blocks = 1;

	if isempty(type)
		type = 'fourier-subsample';
	end

	flt.type = type;

	N = size(f,1);
	M = size(f,2);

	switch type
	case 'fourier'
		flt.flt = f;

	case 'space'
		if M > 1
			f = ifft2(f);
		else
			f = ifft(f);
		end

		zind1 = find(max(abs(f),[],2)<space_threshold);
		zind2 = find(max(abs(f),[],1)<space_threshold);

		if ~isempty(zind1)
			hsupp1 = max(min(zind1)-1,N-max(zind1));
		else
			hsupp1 = N/2;
		end

		if ~isempty(zind2)
			hsupp2 = max(min(zind2)-1,M-max(zind2));
		else
			hsupp2 = M/2;
		end

		if M > 1
			f = [f(N-hsupp1+1:N,M-hsupp2+1:M) f(N-hsupp1+1:N,1:hsupp2); f(1:hsupp1,M-hsupp2+1:M) f(1:hsupp1,1:hsupp2)];
		else
			f = [f(N-hsupp1+1:N); f(1:hsupp1)];
		end
		flt.flt = f;

	case 'fourier-subsample'
		flt.flt0 = f;
		flt.N = N;
		flt.M = M;
		f0 = f;
		for k = 0:log2(N)
			K = 2^k;

			if N/K>floor(N/K)+1e-6
				break;
			end
			
			if M == 1
				f = reshape(f0,[N/K,1,K]);
			else
				f = reshape(f0,[N/K,K,M/K,K]);
				f = permute(f,[1 3 2 4]);
				f = reshape(f,[N/K,M/K,K*K]);
			end
		
			if ~sort_blocks
				flt.inds{k+1} = find(squeeze(max(max(abs(f),[],1),[],2))>fourier_threshold);
			else		
				[temp,inds] = sort(squeeze(max(max(abs(f),[],1),[],2)),'descend');
				flt.inds{k+1} = inds(1:min(max_blocks,length(inds)));
			end
			
			flt.flt{k+1} = f(:,:,flt.inds{k+1});
		end

		if M > 1 && N < 2^6 || (M == 1 && N < 2^11)
			flt.do_periodic = false(size(flt.flt));
		elseif M > 1 && N >= 2^6
			flt.do_periodic = true(size(flt.flt));
			flt.do_periodic(1:min(length(flt.do_periodic),2)) = false;
		elseif M == 1 && N >= 2^11
			flt.do_periodic = true(size(flt.flt));
			flt.do_periodic(1) = false;
		end

	otherwise
		error('Unsupported convolution type.');
	end
end
