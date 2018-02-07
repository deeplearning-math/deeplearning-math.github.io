function [z] = display_angular_profile(in,r)
% display the radial profile of a littlewood-paley
if isnumeric(in)
   littlewood=in;
   if ~exist('r','var')
      [~, r]=max(littlewood(:,1));
end
[N M]=size(littlewood);
[x,y] = meshgrid(-N/2:N/2-1,-M/2:M/2-1);
K=1024;
theta=(1:K)/(K+1)*2*pi;
xi = r * cos(theta);
yi = r * sin(theta);
littlewoodc=fftshift(littlewood);
z=interp2(x,y,littlewoodc,xi,yi);
plot(theta,z);

elseif isstruct(in)
    filters=in;
    lw=littlewood_paley(filters);
    if ~exist('r','var')
       [~, r]=max(lw(:,1));
    end
    hold on;
    for j=1:numel(filters.psi{1})
        for th=1:numel(filters.psi{1}{1})
            display_angular_profile(1/2*abs(filters.psi{1}{j}{th}).^2,r);
        end
    end
    display_angular_profile(abs(filters.phi{1}).^2,r);
    display_angular_profile(lw,r);
    hold off;
end


