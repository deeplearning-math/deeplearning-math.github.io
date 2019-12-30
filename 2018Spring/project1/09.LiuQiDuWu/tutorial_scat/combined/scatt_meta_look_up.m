function lu =scatt_meta_look_up(meta)
    for p=2:numel(meta.order)
        lu(meta.order(p),meta.scale(p)+1,meta.orientation(p)+1)=p;
    end
end