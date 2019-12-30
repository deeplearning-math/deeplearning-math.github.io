%function spl= spline1d(N,s);
N=1000;
NTh=4;
S=4;
itheta=1;
for theta=(0:2*NTh-1) * pi/NTh;
theta
omega=mod(pi + (-N/2:N/2-1) / (N/2)*pi + theta ,2*pi)-pi;
omega=fftshift(omega);
omega=S*omega;
S8=( 5 + 30*cos(omega/2).^2 + 30*sin(omega/2).^2.*cos(omega/2).^2 + 70*cos(omega/2).^4 + 2*sin(omega/2).^4.*cos(omega/2).^2 + 2/3*sin(omega/2).^6) ./...
   (105 * 2^8*sin(omega/2).^8 );
splf=1./(omega.^4.*sqrt(S8));
%splf(1)=1;
splf(isnan(splf))=1;
splfst(itheta,:)=splf;
itheta=itheta+1;
end
plot(sum(splfst.^2,1))
