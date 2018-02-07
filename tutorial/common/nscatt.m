function [out,outo,meta1,meta2]=nscatt(in,options)
%normalized scattering

options.format='split';

%normalize input in L1
in=in/sum(abs(in(:)));
delta=zeros(size(in));
delta(1)=1;
delta=fftshift(delta);


[trans1,meta1]=scatt(in,options);
[trans2,meta2]=scatt(delta,options);

outo=meta1.onorm./meta2.onorm;
out=meta1.norm./meta2.norm;





