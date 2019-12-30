function [z] = display_radial_profile(in,theta)
% display the radial plofile of a littlewood paley in direction theta 

if ~exist('theta','var')
   theta=0;
end

if isnumeric(in)
   littlewood=in;
   [N M]=size(littlewood);
   [x,y] = meshgrid(-N/2:N/2-1,-M/2:M/2-1);
   K=1024;
   r=(-K/2+2:K/2-2)/(K/2)*N/2;
   xi = r * cos(theta);
   yi = r * sin(theta);
   littlewoodc=fftshift(littlewood);
   z=interp2(x,y,littlewoodc,xi,yi);
   plot(r,z);

elseif isstruct(in)
    filters=in;
    hold on;
    for j=1:numel(filters.psi{1})
        for th=1:numel(filters.psi{1}{1})
            display_radial_profile(1/2*abs(filters.psi{1}{j}{th}).^2,theta);
        end
    end
    display_radial_profile(abs(filters.phi{1}).^2,theta);
    lw=littlewood_paley(filters);
    display_radial_profile(lw,theta);
    hold off;
end


