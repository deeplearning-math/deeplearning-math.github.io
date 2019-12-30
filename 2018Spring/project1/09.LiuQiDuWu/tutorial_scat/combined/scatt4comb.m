function [out,meta]=scatt4comb(f,options)
% wrapper for spatial scattering to be used with combined scattering
% take the 2d scattering 
% pading
% put it in a 3d format (2first d spatial, last path)
% unpading
% margin protection

options.null = 1;


% force user to specify J
J = options.J;
a = getoptions(options,'a',2);

%%
% mirror pading
scattMirrorMargin=getoptions(options,'scattMirrorMargin', floor(3*a^J));
scattMirrorMargin =  min(floor(size(f)/2), scattMirrorMargin);
fm = mirrorize(f, scattMirrorMargin);

% scattering
[transf, newscattoptions] = newscatt(fm, options);

% rasterization
outm = zeros(size(transf{1}{1}.signal, 1), ...
    size(transf{1}{1}.signal, 2), ...
    number_of_paths(transf));
raster = 1;
for s = 1:numel(transf)
    for ss = 1:numel(transf{s})
        outm(:,:,raster) = transf{s}{ss}.signal;
        meta.order(raster) = s;
        meta.scale(raster) = transf{s}{ss}.meta.scale;
        meta.orientation(raster) = transf{s}{ss}.meta.orientation;
        raster = raster+1;
    end
end
%%
% unpad + margin protection
ds = round(size(fm,1)/size(outm,1));
unpadMargin = floor(scattMirrorMargin/ds);
scattBorderMargin = getoptions(options, 'scattBorderMargin', 2);
margin = scattBorderMargin + unpadMargin;
out = outm (1+margin(1):end-margin(1), 1+margin(2):end-margin(2), :);

end