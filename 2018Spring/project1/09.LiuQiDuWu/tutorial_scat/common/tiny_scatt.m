
function [out,out2]=tiny_scatt(in, options, j0,J)


M=getoptions(options,'M',2)+0;
aa=getoptions(options,'aa_psi',1);
%downsampling_fac = @(res,j) max(0,floor(j * 1-res-aa));
%downsampling_fac = @(res,j) 0;
filters = options.filters;
filter_type = getoptions(filters,'type','dyadic');
delta=getoptions(options,'delta',1/log2(filters.a));
if strcmp(filter_type,'nondyadic-1d')				% non-dyadic audio filter bank (constant-Q)
	next_bands=getoptions(options,'next_bands',@(j) (audio_next_bands(j,filters.Q,filters.a,filters.J,filters.P,filters.a^(1/log2(filters.a)-delta))));
else
	next_bands=getoptions(options,'next_bands',@(j) (max(0,j+delta)));
end

%  - downsampling_fac
if strcmp(filter_type,'nondyadic-1d')				% non-dyadic audio filter bank (constant-Q)
	downsampling_fac_psi = getoptions(options,'downsampling_fac_psi',@(res,j)(max(0,audio_downsampling(j,filters.Q,filters.a,filters.J,filters.P,2^aa)-res)));
else
	downsampling_fac_psi = getoptions(options,'downsampling_fac_psi',@(res,j) max(0,floor(j * log2(filters.a)-res-aa)));
end
downsampling_fac = downsampling_fac_psi;

temp{1}{1}.signal=in;
temp{1}{1}.scale=j0-1;
%temp{1}{1}.res=max(0,j0-aa);
temp{1}{1}.res=downsampling_fac_psi(0,j0);
temp{1}{1}.fullscale = max(0,temp{1}{1}.scale);
out=2^(temp{1}{1}.res/2)*sum(in);
out2=2^(0*temp{1}{1}.res/1)*sum(in.^2);
%mout.order(1)=0;
%mout.scale(1)=0;
%fullout(:,1)=in(:);

for m=1:M-1
rast=1;
if size(temp,2)<m
break;
end
S=size(temp{m},2);
for s=1:S
curr_resol=temp{m}{s}.res;
tmp=fft(temp{m}{s}.signal);
%for j=temp{m}{s}.scale+1:J-1
for j=next_bands(temp{m}{s}.scale):length(options.filters.psi{curr_resol+1})-1
	ds = downsampling_fac(curr_resol,j);
	%temp{m+1}{rast}.signal=circshift(abs(sub_conv(temp{m}{s}.signal,tmp,options.filters.psi{curr_resol+1}{j+1}{1},2^ds)),-2^(j-ds));
	temp{m+1}{rast}.signal=(abs(sub_conv(temp{m}{s}.signal,tmp,options.filters.psi{curr_resol+1}{j+1}{1},2^ds)));
	temp{m+1}{rast}.res = curr_resol+ds;
	temp{m+1}{rast}.scale=j;
	temp{m+1}{rast}.fullscale = J*temp{m}{s}.fullscale + j;
	rfactor=2^(temp{m+1}{rast}.res/2);
	out=[out rfactor*sum(temp{m+1}{rast}.signal(:))];
	out2=[out2 (1+0*rfactor^2)*sum(temp{m+1}{rast}.signal(:).^2)];
	%out=[out mean(temp{m+1}{rast}.signal(:))];
	%fullout=[fullout (temp{m+1}{rast}.signal(:))];
	%mout.order = [mout.order m];
	%mout.scale = [mout.scale temp{m+1}{rast}.fullscale];
	rast=rast+1;
end
end
end

out2=out2*numel(in);

add_2nd_moment=getoptions(options,'add_2nd_moment',0);
if add_2nd_moment
out=[out (out2.^(1/2))];
end

end





