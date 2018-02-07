function outoptions=configure_wavelet_options(options)

outoptions=options;

unidim=getoptions(options,'unidim',0);

if unidim
family=getoptions(options,'wavelet_family','spline1d');
	switch lower(family)
		case 'cauchy' 
    	outoptions.filter_bank_name=@cauchy_filter_bank;
		otherwise
    	outoptions.filter_bank_name=@spline_filter_bank;
	end
outoptions.wavelet_family=family;
	return
end

family=getoptions(options,'wavelet_family','morlet');
switch lower(family)
  case 'spline'
    outoptions.filter_bank_name=@radial_filter_bank;
    outoptions.cubicspline=1;
    case 'gabor'
    outoptions.filter_bank_name=@gabor_filter_bank_2d;
    outoptions.gab_type='gabor';
    case 'morlet'
    outoptions.filter_bank_name=@gabor_filter_bank_2d;
    outoptions.gab_type='morlet';
    case 'cauchy'
    outoptions.filter_bank_name=@radial_filter_bank;
    outoptions.cubicspline=0;
end
outoptions.wavelet_family=family;
