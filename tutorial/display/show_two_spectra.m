function show_two_spectra(test,out,wavefig,nosquare,sizesignal)


		if nargin < 5
			nosquare=2;
		end

figure;
cc=colormap(gray);
invgray=1-cc;
subplot(1,3,1)
imagesc(crop(test,sizesignal,size(test,1)/2));colormap gray;%axis off;
set(gca,'XTick',[])
set(gca,'YTick',[])


cropfactor=0.75;

subplot(1,3,3)
%out=prepare_logpolar(out);
imagesc(crop(abs(out).^nosquare,round(cropfactor*size(out,1)),round(size(out,1)/2)));colormap(invgray);%axis off;
set(gca,'XTick',[])
set(gca,'YTick',[])

subplot(1,3,2)
spectr=fftshift(abs(fft2(test)));
%spectr=prepare_logpolar(spectr);
imagesc(crop(spectr.^nosquare, round(cropfactor*size(spectr,1)), round(size(spectr,1)/2)));colormap(invgray);%axis off;
set(gca,'XTick',[])
set(gca,'YTick',[])

%wavefig=[path,'wavelets.tif'];
%saveas(gcf,wavefig);
set(gcf,'PaperPositionMode','manual')
set(gcf,'PaperPosition',[0 10 40 10])
print(gcf,'-dtiff',wavefig);

%close


end


function out=prepare_logpolar(in)

rho=1.02;
out=in;
[S1,S2]=size(out);
ix=-S2/2+1:S2/2;
iix=ones(S1,1)*ix;
iy=-S1/2+1:S1/2;
iiy=iy'*ones(1,S2);
r=(iix.^2+iiy.^2);
mask1=(r>=S1^2/4).*(r<rho*S1^2/4);
mask2=(r>=rho*S1^2/4);
out=out.*(1-mask2);
out=max(out(:))*mask1+out.*(1-mask1);
%out=abs(out(:,floor(S2/2):end));

end

