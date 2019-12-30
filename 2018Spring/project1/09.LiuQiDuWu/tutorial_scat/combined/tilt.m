function tf=tilt(f,t)
%f=lena;
%f=f(1:300,1:400);
%tf=tilt(f,t)
% smooth f and subsample along horizontal dir factor t
[N,M]=size(f);
sigma0=1*sqrt(t^2-1);
g=gabor_2d(1,M,sigma0,1,0,0);
g=g./sum(g);
g=repmat(g,N,1);
fsmoothed=ifft(fft(g,[],2).*fft(f,[],2),[],2);


[X,Y]=meshgrid(1:M,1:N);

[xi,yi]=meshgrid(1:t:M,1:N);

tf=interp2(X,Y,f,xi,yi);
