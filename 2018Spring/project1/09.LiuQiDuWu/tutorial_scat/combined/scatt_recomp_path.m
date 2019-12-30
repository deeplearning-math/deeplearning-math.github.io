function [p,scale,orientation]=scatt_recomp_path(orientations,scales,metalookup,L,J)
mp=numel(orientations);
orientation=0;
scale=0;
for k=1:mp
    %orientations(k)=floor(meta.orientation(p)/L^(mp-k));
    %scales(k)=floor(meta.scale(p)/(J)^(mp-k));
    orientation=orientation+orientations(k)*L^(mp-k);
    scale=scale+scales(k)*J^(mp-k);
end
if (scale+1<=size(metalookup,2))
    p=metalookup(mp+1,scale+1,orientation+1);
else
    p=0;%not a valid path
end
end