function [type,sparsenorms]=compare_dct_strategies(in,options)


%original scattering coefficients
opts=options;
opts.dct=0;
opts.mirror=0;

%orig=newscatt(in,opts);

orig=scatt(in,opts);
orig=reshape(orig,size(orig,1)*size(orig,2),numel(orig)/(size(orig,1)*size(orig,2)));

sparsenorms(:,1)=sum(abs(orig),2);

%transform along scale only
opts.dct=1;
opts.dct_transf_mode=2;

raster=2;
for flip=1:1
	for diff=0:1
			opts.flip_order=flip;
			opts.differential_dct_sc=diff;
			transf=scatt(in,opts);
			sparsenorms(:,raster)=sum(abs(transf),2);
			raster=raster+1;
	end
end

[nothing,type]=min(sparsenorms,[],2);






