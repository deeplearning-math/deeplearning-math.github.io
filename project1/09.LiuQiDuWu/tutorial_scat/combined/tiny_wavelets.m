function filters=tiny_wavelets(N,L,display)
% tiny_wav_t compute morlet wavelets with very small support
% these wavelet will be used as 'orbital' wavelet
% for the pinwheel scattering
% parameters are :
% N number of sample (16)
% L maximum scale
% display = display the wavelet or not
% Wavelets are stored in fourier domain
if ~exist('display','var')
    display=0;
end
% compute a tiny wavelet transform (on N=16 samples)
sigma0=0.7;
xi0=3*pi/4;
slant=1;

%compute wavelet on large support and periodize them to avoid
%boundary effects
K=16;
NN=K*N;
for l=0:L-1
    %normalization scale since morlet_2d are normalized for 2d
    scale=2^l;
    filterlarge=scale*2*sqrt(2*pi*sigma0^2/slant)*morlet_2d(1,NN,sigma0*scale,slant,xi0/scale,0);
    filter=zeros(1,N);
    for k=1:K
        filter=filter+filterlarge((1:N)+(k-1)*N);
    end
    if display
        
        subplot(L+1,1,l+1);
        plot(1:N,real(filter),1:N,imag(filter));
        
    end
    %sum(filter)
    filters.psi{l+1}=conj(fft(filter));
end

scale=2^L;

filterlarge=scale*sqrt(2*pi*sigma0^2/slant)*gabor_2d(1,NN,sigma0*scale,slant,0,0);
filter=zeros(1,N);
for k=1:K
    filter=filter+filterlarge((1:N)+(k-1)*N);
end
if display
    subplot(L+1,1,L+1);
    plot(real(filter));
end
filters.phi=conj(fft(filter));
end