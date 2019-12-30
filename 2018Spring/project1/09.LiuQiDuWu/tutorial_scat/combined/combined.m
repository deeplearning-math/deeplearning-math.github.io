function [outmat,metaf] = combined(scattvectors,meta,options)
% This function computes rotation combined scattering from spatial
% scattering.
% Scattering must have been computed with 8 (or 16) orientations.
% each line of scattvectors is a scattering vector
% meta contains output of scatt.m
% The following parameters must be 

%%

options.null=1;
combprc = combined_precomputation(meta,options);
orbits=combprc.orbits ;
Jc=combprc.Jc;
tw=combprc.tw;
L=combprc.L;

%%
norb=size(combprc.orbits,1);
nsig=size(scattvectors,1) ;
sizeorb=size(combprc.orbits,2);
orbitalconcatenation=zeros(norb*nsig,sizeorb);
for i=1:nsig
    cursig=scattvectors(i,:);
    orbitalconcatenation((i-1)*norb+(1:norb),:)= cursig(orbits);    
end

Mc=getoptions(options,'Mc',min(Jc,2)); % maximum order of combined scattering (between 1 and 3)
SSJ{1}{1}=orbitalconcatenation;
metac{1}{1}=0;

% apply wavelet modulus operator along orbits (and iterate over it)
for m=1:Mc
    nextp=1;
    for p=1:numel(metac{m})
        sigf=fft(SSJ{m}{p},[],2);
        scales=metac{m}{p};
        
        for j=1:Jc
            if j>scales(end)
                Wf=repmat(tw.psi{j},size(sigf,1),1);
                SSJ{m+1}{nextp}=abs(ifft(Wf.*sigf,[],2));
                metac{m+1}{nextp}=[scales,j];
                nextp=nextp+1;
            end
        end
    end
end

% average and subsample along arbits
sub=min(L,2^(Jc));
poto=0:L-1;
potof=fft(0:L-1);
for m=1:Mc+1
    for p=1:numel(metac{m})
        sigf=fft(SSJ{m}{p},[],2);
        %ifft(conj(tw.phi))
        Wf=repmat(tw.phi,size(sigf,1),1);
        SLSJ{m}{p}=abs(ifft(Wf.*sigf,[],2));
        SLSJ{m}{p}=sqrt(sub)*SLSJ{m}{p}(:,1:sub:end);
        
        
        positionOnTheOrbit{m}{p}=poto(1:sub:end);
        %positionOnTheOrbitSmoothed{m}{p}=ifft(Wf(1,:).*potof,[],2);
        %positionOnTheOrbitSmoothed{m}{p}=positionOnTheOrbitSmoothed{m}{p}(1,1:sub:end);
        %positionOnTheOrbitSmoothed{m}{p}=floor(positionOnTheOrbitSmoothed{m}{p});
        
    end
end
sizeorb2=size(SLSJ{1}{1},2);

s=vectorize2(SLSJ);


positionOnTheOrbit=vectorize2(positionOnTheOrbit);
positionOnTheOrbit=repmat(positionOnTheOrbit,1,norb);
%positionOnTheOrbitSmoothed=vectorize2(positionOnTheOrbitSmoothed);
%positionOnTheOrbitSmoothed=repmat(positionOnTheOrbitSmoothed,1,norb);

%s=SLSJ{1}{1};
%for m=2:Mc
%sm=vectorize(SLSJ{m});

%s=cat(3,s,sm);

%d

metaf.orientations=repmat(meta.orientation(orbits(:,1))',[1,size(s,2)]);
metaf.scales=repmat(meta.scale(orbits(:,1))',[1,size(s,2)]);
metaf.orderspatial=repmat(meta.order(orbits(:,1))',[1,size(s,2)]);
norb2=numel(metaf.orderspatial);
metaf.orientations=reshapebis(metaf.orientations,1,norb2);
metaf.scales=reshapebis(metaf.scales,1,norb2);
metaf.orderspatial=reshapebis(metaf.orderspatial,1,norb2);

%scalerot(1)=0;
%orderrot(1)=1;
%currind=2;
for m=1:Mc
    
    for pcombined=1:numel(metac{m})
        %scalerot(currind) = 0;
        %orderrot(currind) = m;
        scalerot{m}{pcombined}=0;
        orderrot{m}{pcombined}=m * ones(1,sizeorb2);
        for ii=2:numel(metac{m}{pcombined})
            scalerot{m}{pcombined}=scalerot{m}{pcombined}...
                + (metac{m}{pcombined}(ii)-1)*Jc^(m-ii);
            %polynomial coding of scale of combined scatt
            %(\tilde{j_1}...\tilde{j_\tilde{p}}) in ESANN 12 PAPER
        end
        
        scalerot{m}{pcombined}=scalerot{m}{pcombined}*ones(1,sizeorb2);
    end
end

scalerot=vectorize2(scalerot);
scalerot=repmat(scalerot,1,norb);
orderrot=vectorize2(orderrot);
orderrot=repmat(orderrot,1,norb);


metaf.rotationOrder=orderrot;
metaf.rotationPathVariable=scalerot;
%metaf.positionOnTheRotationOrbitSmoothed=positionOnTheOrbitSmoothed;
metaf.positionOnTheRotationOrbit=positionOnTheOrbit;

%rearrange coefficients in line
%outmat=s;

outmat=zeros(nsig,norb2);
for i=1:nsig
    outmat(i,:)=reshapebis(s((i-1)*norb+(1:norb),:),1,norb2);
end


