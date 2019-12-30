function g= gaussian(x,options)
sigma=getoptions(options,'gaussian_sigma',0.5); % good tradeoff for 4 orientations
g=exp(-x.^2 /(2*sigma^2) );
end
