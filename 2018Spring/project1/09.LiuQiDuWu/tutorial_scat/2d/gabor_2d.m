function [gab] = gabor_2d(N,M,sigma0,slant,xi,theta,offset)
%function [gab] = gabor_2d(N,M,sigma0,slant,xi,theta)
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
A=inv(Rth) * [1/sigma0^2, 0 ; 0 slant^2/sigma0^2] * Rth ;
s=x.* ( A(1,1)*x + A(1,2)*y) + y.*(A(2,1)*x + A(2,2)*y ) ;
%normalize sucht that the maximum of fourier modulus is 1
gabc=exp( - s/2 + 1i*(x*xi*cos(theta) + y*xi*sin(theta)));
gab=1/(2*pi*sigma0*sigma0/slant)*fftshift(gabc);

end
