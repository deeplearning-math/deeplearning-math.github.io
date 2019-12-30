%we start with L1 signals of compact support
clear options;

N=4096/2;
Q=256;
sigma=2;

if exist('signal')~=1
signal=zeros(N,1);
signal(N/2-Q:N/2+Q)=1;
signal=signal.*randn(size(signal));
signal=gabor1dfilter(signal,sigma);
end

options.J=11;
options.M=5;
options.delta=-4;
options.aa_psi=100; %no downsampling in the intermediate cascade
options.renorm_study=1;

[out,meta]=scatt(signal,options);

size(out)


%study last order ratios
evolution=zeros(size(meta.order));
scaledecr=zeros(size(meta.order));
isprog=zeros(size(meta.order));
last=find(meta.order==max(meta.order));
for l=last
	parent=find((meta.scale==floor(meta.scale(l)/J))&(meta.order==max(meta.order)-1));
	if(meta.norm_ratio_orig(l) > meta.norm_ratio_orig(parent))
		evolution(l)=1;
	else
		evolution(l)=-1;
	end
	if(mod(meta.scale(l),J) > mod(meta.scale(parent),J))
		scaledecr(l)=1;
	else
		scaledecr(l)=-1;
	end
	isprog(l)=isprogressive(meta.scale(l),J,max(meta.order)-1);
end



