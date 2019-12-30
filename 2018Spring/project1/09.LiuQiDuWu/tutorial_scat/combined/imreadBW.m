function [f] = imreadBW (filename)
% 24/01/2011
% This function reads any image, convert it in black and white and rescale
% it to range 0..1
fcol= imread(filename);
fcold=double(fcol);
if (size(size(fcol),2)==3)
	f=1/255*(0.3*fcold(:,:,1) + 0.59*fcold(:,:,2) + 0.11*fcold(:,:,3));
end
if (size(size(fcol),2)==2)
	f=fcold;
end
end