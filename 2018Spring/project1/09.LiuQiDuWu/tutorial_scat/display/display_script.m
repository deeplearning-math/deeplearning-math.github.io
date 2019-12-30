
clear all;
close all;
startup

options.J=6;
options.M=2;
options.delta=0;
options.aa_psi=2;

T=25;
TT=2;

tr=retrieve_cure_data(TT,1,T);

T=size(tr,2);
raster=1;

for t=1:T
	for tt=1:size(tr{t},2)
		[out,meta{raster}]=scatt(tr{t}{tt},options);
		averages(raster,:)=meta{raster}.sq_scatt_ave;
		%averages(raster,:)=meta{raster}.scatt_ave;
		raster=raster+1;
	end
	fprintf('done class %d \n',t)
end
aver=mean(averages);


[im1,im2,im3]=display_2d_split(meta{1},aver);


