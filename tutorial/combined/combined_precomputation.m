function combprc = combined_precomputation(meta,options)
metalookup =scatt_meta_look_up(meta);
%L=optionsfilters.L;
J=max(meta.scale(meta.order==2))+1;
L=max(meta.orientation(meta.order==2))+1;
Jc=options.Jc;% scale of combined scattering
rotwav=getoptions(options,'rotwav','gab');
switch rotwav
    case 'gab'
        tw=tiny_wavelets(L,Jc,0);
    case 'haar'
        tw=tiny_wavelets_haar(L,Jc,0);
end
is_not_yet_in_an_orbit=ones(1,numel(meta.order));
%J=optionsfilters.J;
iorb=1;
for p=2:numel(meta.order)
    if (is_not_yet_in_an_orbit(p))
        %decompose p
        [orientations,scales]=scatt_decomp_path(p,meta,L,J);
        
        
        for alpha=0:L-1
            %incr orientations
            orientations2=mod(orientations+alpha,L);
            
            %recompose p2
            p2=scatt_recomp_path(orientations2,scales,metalookup,L,J);
            
            %store the indice
            orbits(iorb,alpha+1)=p2;
            is_not_yet_in_an_orbit(p2)=0;
        end
        iorb=iorb+1;
    end
end
combprc.orbits=orbits;
combprc.tw=tw;
combprc.Jc=Jc;
combprc.L=L;