function cau=cauchyf(omega,options);

lambda=getoptions(options,'cauchy_lambda',0.01);
% lamda control the maximum discontinuity of the filter
p=getoptions(options,'cauchy_p',2);
% p control the selectivity of the filter

epsilon=lambda^(1/p) / exp(1);
m=-log(epsilon) + log(-log(epsilon));
sc=m/(2*pi);


cau= (sc* omega).^p .* exp( - p*(sc*omega-1));
