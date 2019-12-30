N=512;
options.J=6;
options.M=2;
%options.filter_bank_name=@gabor_filter_bank_2d;
options.wavelet_family='spline';
options.L=8;
options.renorm_study=1;

raster=1;
%gabor functions
L=options.L;

theta=pi/4;
xi=3*pi/6;
slant=1;
sigma=10;

alpha=pi/(12*L);
dil=.95;
theta1=theta+alpha/2;
theta2=theta-alpha/2;

Ldef=[[1-dil*cos(alpha) dil*sin(alpha)];[-dil*sin(alpha) 1-dil*cos(alpha)]];
normdef=max(svd(Ldef));

if 1
%indicator of a rectangle
%test{raster}=zeros(N);
%M=10;
%test{raster}(N/2-M/2:N/2+M/2,N/2-M/2:N/2+M/2)=1;
%[out{raster},meme{raster}]=logpolar_scattering(test{raster},options,0);
%show_two_spectra(test{raster},out{raster},fullfile(mpath,sprintf('example-%d.tif',raster)),.5,4*M);
%raster=raster+1;
%
%gabor
XX=randn(N);
%test{raster}=(real(ifft2(fft2(gabor_2d(N,N,sigma,slant,xi,theta1)).*fft2(XX))));
test{raster}=fftshift(real(gabor_2d(N,N,sigma,slant,xi,theta1)));
[out{raster},meme{raster}]=logpolar_scattering(test{raster},options,0);
show_two_spectra(test{raster},out{raster},fullfile(mpath,sprintf('example-%d.tif',raster)),.5,12*sigma);
raster=raster+1;


%test{raster}=(real(ifft2(fft2(gabor_2d(N,N,sigma/dil,slant,xi*dil,theta1-alpha)).*fft2(XX))));
test{raster}=fftshift(real(gabor_2d(N,N,sigma/dil,slant,xi*dil,theta1-alpha)));
[out{raster},meme{raster}]=logpolar_scattering(test{raster},options,0);
show_two_spectra(test{raster},out{raster},fullfile(mpath,sprintf('example-%d.tif',raster)),.5,12*sigma);
raster=raster+1;

test{raster}=fftshift(real(gabor_2d(N,N,sigma/(dil*dil),slant,xi*dil*dil,theta1-2*alpha)));
[out{raster},meme{raster}]=logpolar_scattering(test{raster},options,0);
show_two_spectra(test{raster},out{raster},fullfile(mpath,sprintf('example-%d.tif',raster)),.5,12*sigma);
raster=raster+1;

%compute ratios
f1=test{1}(:);
f2=test{2}(:);
sc1=meme{1}(:);
sc2=meme{2}(:);
tmp=abs(fft2(test{1}));
fm1=tmp(:)*(norm(f1)/norm(tmp(:)));
tmp=abs(fft2(test{2}));
fm2=tmp(:)*(norm(f2)/norm(tmp(:)));

A=norm(sc1-sc2)/(normdef*(norm(f1)+norm(f2))*.5)
B=norm(fm1-fm2)/(normdef*(norm(f1)+norm(f2))*.5)

test{raster}=double(rgb2gray(imread('4.2.03.tiff')));
test{raster}(end,:)=test{raster}(end-1,:);
test{raster}=test{raster}-mean(test{raster}(:));
[out{raster},meme{raster}]=logpolar_scattering(test{raster},options,0);
show_two_spectra(test{raster},out{raster},fullfile(mpath,sprintf('example-%d.tif',raster)),.5,512);
raster=raster+1;
test{raster}=sphere_warping(test{raster-1},4,0,16);
[out{raster},meme{raster}]=logpolar_scattering(test{raster},options,0);
show_two_spectra(test{raster},out{raster},fullfile(mpath,sprintf('example-%d.tif',raster)),.5,512);
raster=raster+1;


%compute ratios
f1=test{3}(:);
f2=test{4}(:);
sc1=meme{3}(:);
sc2=meme{4}(:);
tmp=abs(fft2(test{3}));
fm1=tmp(:)*(norm(f1)/norm(tmp(:)));
tmp=abs(fft2(test{4}));
fm2=tmp(:)*(norm(f2)/norm(tmp(:)));

A=norm(sc1-sc2)/(normdef*(norm(f1)+norm(f2))*.5)
B=norm(fm1-fm2)/(normdef*(norm(f1)+norm(f2))*.5)




if 0
%gabor interference
test{raster}=fftshift(real(gabor_2d(N,N,1*sigma,slant,xi,theta1)+gabor_2d(N,N,sigma/dil,slant,xi*dil,theta1+alpha)));
[out{raster},meme{raster}]=logpolar_scattering(test{raster},options,0);
show_two_spectra(test{raster},out{raster},fullfile(mpath,sprintf('example-%d.tif',raster)),.5,12*sigma);
raster=raster+1;
end

end
if 0

%sparsity analysis
%lec=double(rgb2gray(imread('lena.ppm')));
tmp=zeros(4*N);

ix=1:4*N;
iix=ones(4*N,1)*ix;
iiy=iix';
rho=1;
tmp=double(iix < rho*iiy);

apert=pi/20;
angles=angle(iix+i*iiy);
radius=sqrt(iix.^2+iiy.^2);

tmp=double(angles < pi/4+apert).*(angles > pi/4-apert).*(radius < 3*N);
tmp=circshift(tmp,[N/2 N/2]);
gg=fspecial('gaussian',[9 9],1);
tmp=imfilter(tmp,gg);
[gr1,gr2]=gradient(tmp);
tmp=sqrt(gr1.^2+gr2.^2);

gg=fspecial('gaussian',[9 9],4);
tmp=imfilter(tmp,gg);
test{raster}=tmp(1:4:end,1:4:end);


options=configure_wavelet_options(options)
filter_bank=getoptions(options,'filter_bank_name',@radial_filter_bank);
filters=filter_bank(size(test{raster}),options);

psi=filters.psi{1};
phi=filters.phi{1};
lp=littlewood_paley(filters);

test{raster}=ifft2(fft2(test{raster}).*(sqrt(lp{1})));


if 0
if 0
r=(iix-N/2).^2+(iiy-N/2).^2;
r=max(0,N/2-(r).^.5);
[tmpg1,tmpg2]=gradient(test{raster});
test{raster}=sqrt(tmpg1.^2+tmpg2.^2);
test{raster}=test{raster}.*r;
gg=fspecial('gaussian',[9 9],1);
test{raster}=imfilter(test{raster},gg);
else
gg=fspecial('gaussian',[9 9],0.2);
test{raster}=imfilter(test{raster},gg);
mask=zeros(N);
pad=0;
mask(pad+1:end-pad,pad+1:end-pad)=1;
gg=fspecial('gaussian',[9 9],3);
mask=imfilter(mask,gg);
mask=mask/max(mask(:));
test{raster}=test{raster}.*mask;
end
end

tempo=randn(size(test{raster}));
test{raster}=test{raster}-mean(test{raster}(:));
test{raster}=test{raster}/norm(test{raster}(:));
tempo=tempo-mean(tempo(:));
tempo=tempo/norm(tempo(:));
[geq,l1f,l1g,l1eq]=equalize_first_order_scattering(test{raster},tempo,options,psi,phi,lp);

	[res,meta]=scatt(test{raster},options);
	fprintf('scatt done \n')
	%[out{raster}]=fulldisplay2d(meta.ave,meta,0,0,1);
	[out_split]=fulldisplay2d(meta.ave,meta,0,2,1);

	raster=raster+1;

	test{raster}=geq;
	[res,meta]=scatt(test{raster},options);
	fprintf('scatt done \n')
	%[out{raster}]=fulldisplay2d(meta.ave,meta,0,0,1);
	[out_split_bis]=fulldisplay2d(meta.ave,meta,0,2,1);
	normvalues(1)=max(max(out_split{1}(:)),max(out_split_bis{1}(:)));
	normvalues(2)=max(max(out_split{2}(:)),max(out_split_bis{2}(:)));
	show_two_spectra_dlux(test{raster-1},out_split, fullfile(mpath,sprintf('example-%d.tif',raster-1)),.5,normvalues);
	show_two_spectra_dlux(test{raster},out_split_bis, fullfile(mpath,sprintf('example-%d.tif',raster)),.5,normvalues);

	raster=raster+1;


end



%
%if 0
%
%%synthetic textures
%
%test{raster}=randn(N);
%out{raster}=logpolar_scattering(test{raster},options);
%show_two_spectra(test{raster},out{raster});
%raster=raster+1;
%
%
%delta=.95;
%test{raster}=double(rand(N)>delta);
%out{raster}=logpolar_scattering(test{raster},options);
%show_two_spectra(test{raster},out{raster});
%raster=raster+1;
%
%
%%Cure textures
%
%test{raster}=trcure{2}{1};
%out{raster}=logpolar_scattering(test{raster},options);
%show_two_spectra(test{raster},out{raster});
%raster=raster+1;
%
%test{raster}=trcure{20}{1};
%out{raster}=logpolar_scattering(test{raster},options);
%show_two_spectra(test{raster},out{raster});
%raster=raster+1;
%%
%
%%Deformed digits
%trmnist=retrieve_mnist_data(10,2);
%test{raster}=zeros(N);
%test{raster}(N/2-15:N/2+16,N/2-15:N/2+16)=trmnist{3}{1};
%out{raster}=logpolar_scattering(test{raster},options);
%show_two_spectra(test{raster},out{raster});
%raster=raster+1;
%
%test{raster}=zeros(N);
%test{raster}(N/2-15:N/2+16,N/2-15:N/2+16)=trmnist{3}{3};
%out{raster}=logpolar_scattering(test{raster},options);
%show_two_spectra(test{raster},out{raster});
%raster=raster+1;
%
%end
