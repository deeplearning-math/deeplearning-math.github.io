function [orientations,scales]=   scatt_decomp_path(p,meta,L,J)
    mp=meta.order(p)-1;
    for k=1:mp
        orientations(k)=mod(floor(meta.orientation(p)/L^(mp-k)),L);
        scales(k)=mod(floor(meta.scale(p)/J^(mp-k)),J);
    end
end