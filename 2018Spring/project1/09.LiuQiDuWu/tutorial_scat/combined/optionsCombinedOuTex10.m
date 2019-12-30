function options = optionsCombinedOuTex10()
% combined scattering options for OUTEX10 
% see tune_gabor.m to see the three requirements on filters for a good
% tuning of parameters.

% ==========  parameters for spatial scattering =================

% parameters for wavelet shape :
options.sigma0 = 0.6957; % width of low pass filter of spatial scattering
options.sigma00 = 0.8506; % width of mother wavelet of spatial scattering
options.xi0 = 3*pi/4; % frequency peak of mother wavelet of spatial scattering
options.slant = 0.5; % slant of mother wavelet of spatial scattering
options.gab_type = 'morlet'; % type of wavelet filter of spatial scattering

% parameters for the algorithm of spatial scattering : 
options.a = 2; % the dilation factor of spatial wavelet
options.L = 8; % number of orientation of wavelet of spatial scattering
options.J = 5; % maximum scale of spatial scattering
options.M = 2; % maximum order of spatial scattering

% parameters for precision : 
options.aa = 1; % the antialiasing for spatial scattering (subsampled rate = max(1,2^(J-aa)))
options.scattBorderMargin = 1; % the margin for mirror padding


% ==========  parameters for combined scattering ==================
options.Jc = 3; % maximum scale of rotation invariance
options.Mc = 2; % maximum order or rotation scattering


% ROTAION INVARIANCE : 
% If 2^Jc = L (the maximum scale for rotation is equal to the number of
% angle of spatial wavelets), then the combined scattering is fully
% rotation invariant.

