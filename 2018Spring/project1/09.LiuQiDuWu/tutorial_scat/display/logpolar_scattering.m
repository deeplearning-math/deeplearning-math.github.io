function [outimage,out]=logpolar_scattering(in,options,type)

		options.renorm_study=1;
		%coco=1-colormap(gray);
		
		[out,meta]=scatt(in,options);
		fprintf('scatt done \n')
		meta.dirac_norm=meta.dirac_ave;
		[outimage,outorder]=fulldisplay2d(meta.ave,meta,0,type,1);

		%figure
		%picture=zeros([size(outimage) 3]);
		%picture(:,:,2)=(outimage/max(outimage(:)));
		%picture(:,:,1)=(outorder/max(outorder(:)));
		%imagesc((outimage))
		%colormap(coco)

		%figure
		%imagesc(fftshift((abs(fft2(in)))))
		%colormap(coco)

		%[outnormal]=fulldisplay2d(out,meta,0,1,0);
		%figure
		%imagesc(outnormal(:,:,1))
		%colormap gray


