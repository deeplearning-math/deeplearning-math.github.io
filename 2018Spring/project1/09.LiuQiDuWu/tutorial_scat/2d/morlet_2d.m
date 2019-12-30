function [gab] = morlet_2d(N,M,sigma,slant,xi,theta,offset)
%function [gab] = gabor_2d(N,M,sigma0,slant,xi,theta)
% N = W
% M = H
% 2d elliptic gabor filter
if ~exist('offset','var')
    offset=[0,0];
end
[x , y]=meshgrid(1:M,1:N);

x=x-ceil(M/2)-1;
y=y-ceil(N/2)-1;
x=x-offset(1);
y=y-offset(2);

Rth=rotationMatrix2d(theta);
A=inv(Rth) * [1/sigma^2, 0 ; 0 slant^2/sigma^2] * Rth ;

s=x.* ( A(1,1)*x + A(1,2)*y) + y.*(A(2,1)*x + A(2,2)*y ) ;
%A=inv(Rth) * [1/sigma, 0 ; 0 slant/sigma] * Rth ;
%s=( A(1,1)*x + A(2,1)*y).^2 + (A(1,1)*x + A(2,2)*y ).^2 ;
%normalize sucht that the maximum of fourier modulus is 1
gabc=exp( - s/2).*( exp(1i*(x*xi*cos(theta) + y*xi*sin(theta)))- exp(-(xi*sigma)^2/2));
gab=1/(2*pi*sigma^2/slant)*fftshift(gabc);

end
