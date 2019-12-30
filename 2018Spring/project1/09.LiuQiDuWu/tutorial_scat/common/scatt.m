function [out,meta,newscattoptions] = scatt(in,options)
%wrapper for the scattering transform, 
%[out,meta] = scatt(in,options)
%options are
% .J = the maximum scale of the transform [default 3]
% .M = the maximum scattering order [default 2]
% .L = the number of different orientations [default 6]
% .format = 'split' for a bidimensional output (bands x spatial coeffs) [default]
% 	'array' for a unidimensional output (bands and spatial coeffs)
%	'raw' to have access to all metadata of the output (path information)			
%	'patch' patch encapsulation (only available for 2d input)
% .downsample = 1 for spatial downsampling at coarse scales to accelerate computations
% .dct  
%
	options.null = 1;

	if sum(size(in)>1)==1 
		meta.unidim=1;
		options.unidim=1;
		if size(in,1)<size(in,2)
			in=in';
		end
	end

	format = getoptions(options,'format','split');

	mirror=getoptions(options,'mirror',0);
	dyadic_padding=getoptions(options,'dyadic_padding',1);
	renorm_study=getoptions(options,'renorm_study',0);
	renorm_scatt=getoptions(options,'renorm_scatt',0);
        options=configure_wavelet_options(options);
	compact_scattering=getoptions(options,'compact_scattering',0);

	
	if dyadic_padding
		powsup=2.^ceil(log2(size(in)));	
		marg=(powsup-size(in))/2;
		in=mirrorize(in,ceil(marg));
	end

	if mirror
		J=getoptions(options,'J',3);
		filterconst=getoptions(options,'filter_padding',3);
		marg=min(floor(max(size(in))/2),2^(J+2)*filterconst);
		marg=[marg marg];
		in=mirrorize(in,ceil(marg));
	end

	[Sp]=size(in);
	%if renorm_study
	%	options.filters=options.filter_bank_name(Sp,options);
	%	lit=littlewood_paley(options.filters);
	%	meta.lp_correction=mean(lit{1}(:))*4/pi;
	%end

	%do the transform
	[transf,newscattoptions] = newscatt(in, options);

    
    
	if renorm_study | (renorm_scatt && (~(isfield(options,'dirac_norm') && ~isempty(options.dirac_norm))))
		dirac=zeros(size(in));
		dirac(1)=1;%sum(abs(in(:)));
		dirac=fftshift(dirac);
		diractransf=newscatt(dirac,options);
	end

	%formatting the output
	[St]=size(transf{1}{1}.signal);

	%Unpadding (eventually put this in a separate function)
	if mirror | dyadic_padding
		margs=floor(marg*St(1)/Sp(1));
		Cinf=(1+margs);%*ones(size(St));
		Csup=St-margs;%*ones(size(St));
	else
		Cinf=ones(size(St));
		Csup=St;
	end
	singletons=find(St==1);
	Cinf(singletons)=1;
	Csup(singletons)=1;

	options.scatdims=[Cinf Csup];

	if strcmp(format,'patch')==1
		out=patch_reformat(transf,options);
		return;
	end

	if strcmp(format,'raw')==1
		out=transf;
		return;
	end

	S=size(transf);
	raster=1;
        out=zeros([Csup-Cinf+1 number_of_paths(transf)]);
	for s=1:S(2)
		SS=size(transf{s});
		for ss=1:SS(2)
                    slice=transf{s}{ss}.signal(options.scatdims(1):options.scatdims(3),options.scatdims(2):options.scatdims(4));
                    out(:,:,raster)=slice;
                    meta.order(raster)=s;
                    meta.scale(raster)=transf{s}{ss}.meta.scale;
                    meta.orientation(raster)=transf{s}{ss}.meta.orientation;
                    meta.scatt_ave(raster)=mean(slice(:));
                    %meta.sq_scatt_ave(raster)=mean(slice(:).^2);
                    meta.norm(raster)=norm(slice(:));
		    meta.ave(raster)=transf{s}{ss}.ave;
                    if renorm_study
			%meta.onorm(raster)=sum(transf{s}{ss}.orig(:));
			meta.ave(raster)=mean(slice(:));
			meta.dirac_norm(raster)=norm(diractransf{s}{ss}.signal(:));
			meta.dirac_ave(raster)=mean(diractransf{s}{ss}.signal(:));
			%meta.dirac_onorm(raster)=norm(diractransf{s}{ss}.orig(:));
			%out(:,:,raster)=slice/meta.dirac_norm(raster);
			meta.norm_ratio(raster)=norm(slice)/meta.dirac_norm(raster);
                    end
                    if renorm_scatt
                      if(exist('diractransf','var'))
                        options.dirac_norm(raster) = norm(diractransf{s}{ss}.signal(:));
                        meta.dirac_norm(raster)=options.dirac_norm(raster);
                      end
                      out(:,:,raster)=slice / options.dirac_norm(raster);
                      meta.norm(raster)= meta.norm(raster) / options.dirac_norm(raster);
                    end
                    raster=raster+1;
		end
	end
        discard_lowpass=getoptions(options,'discard_lowpass',0);
        if discard_lowpass 
        out(:,:,1)=0;
        end
	out=squeeze(out);

     combined_rotation=getoptions(options,'combined_rotation',0);
     if combined_rotation
      tempo=reshape(out,size(out,1)*size(out,2),size(out,3));
      [outc,metc]=combined(tempo,meta,options);
      if getoptions(options,'renorm_comb_scatt',0)
        meta.norm=sqrt(sum(outc.^2));
        outc=outc ./ (ones(size(outc,1),1)*options.dirac_norm);
      end
      out=reshape(outc,size(out,1),size(out,2),size(outc,2));
     end   


    if strcmp(format,'bidim') & length(size(out)==3)
      out=reshape(out,size(out,1)*size(out,2),size(out,3));
    end
   
    delocalized=getoptions(options,'delocalized',0);
    if delocalized
      out=meta.scatt_ave;
      out(2,:)=meta.scatt_ave; 
      meta.norm=meta.scatt_ave;
    end

	if compact_scattering & (max(meta.order)>2) 
		sec_order=find(meta.order==2);
		J=max(meta.scale(sec_order))+1;
		L=max(meta.orientation(sec_order))+1;
		M=max(meta.order)-1;
		nscatcoeffs=min(100,getoptions(options,'nscatcoeffs',50));
                transf_type=getoptions(options,'transf_type','dct');
		
                %transform
		[out,meta]=scatt_orthotransf(out,meta,options);

		%keep selected coefficients
                [rien,sortmask]=sort(meta.combined_scale,'ascend');
                out=out(:,sortmask);
		out=out(:,1:round(nscatcoeffs*length(meta.order)/100));
                meta.order=meta.order(sortmask);
                meta.order=meta.order(1:round(nscatcoeffs*length(meta.order)/100));
                meta.norm=sqrt(sum(out.^2));
      end

    
  whitening=getoptions(options,'whitening',0);
  %we start by renormalizing with the scattering moments
  if whitening
    %tempo=zeros([Csup-Cinf+1 number_of_paths(transf)]);
    tempo=zeros(size(out));
    epsi_white=getoptions(options,'epsi_white',0.00);
    if whitening >= 4
    epsi_white=0;
    end
    switch whitening
     case {1,4} %scattering moments
     for m=1:max(meta.order)
      supp=find(meta.order==m);
      meta.ordecay(m)=sum(meta.norm(supp).^2);
      stemp=size(tempo);
      tempo2=reshape(tempo,numel(tempo)/stemp(end),stemp(end));
      tempo2(:,supp)=meta.ordecay(m)*ones([numel(tempo)/stemp(end) length(supp)]);
      tempo=reshape(tempo2,stemp);
      meta.whitening=tempo(:)';
     end
     case {2,5} %path whitening 
      tmp=ones(numel(tempo)/length(meta.order),1)*(meta.norm.^2+epsi_white*ones(size(meta.norm)));  
      tempo=reshape(tmp,size(tempo));
      meta.whitening=tempo(:)';
     case {3,6} %full whitening
      meta.whitening=out(:).^2+epsi_white;
      meta.whitening=meta.whitening';
    end
  
  end
 
  if strcmp(format,'array')==1
    out=out(:)';
  end
  
  if (isfield(options,'whitening_mask') && ~isempty(options.whitening_mask))
   out=out./options.whitening_mask;
  end




