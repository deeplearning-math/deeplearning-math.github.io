function [out,inmask]=haartransf_evo(in,dim,m,J,L);
%dim goes to 0 to 2*m-1. 
%it indicates the coordinate along which we will 
%transform. it will be used only to record the path
%in the meta information

out=in;
[K,L1]=size(in.coeffs);
[K2,L2]=size(in.mask);

%encode haar transform
if dim < m
%transform along orientations
startpos=mod(floor(in.code_or_pos/L^dim),L);
else
%transform along scales
startpos=mod(floor(in.code_sc_pos/J^(dim-m)),J);
end

npixels=L1/L2;


if 0

inmask=repmat(in.mask,[1 npixels]);;
inweight=repmat(in.weight,[1 npixels]);;
incode=repmat(startpos,[1 npixels]);
coeffs=in.coeffs;
%inmask=[];
%inweight=[];
%incode=[];

%for l=1:npixels
%if(mod(l,320)==31)
%fprintf('h')
%end
%inmask=[inmask in.mask];
%inweight=[inweight in.weight];
%incode=[incode startpos];
%end

else
coeffs=reshape(in.coeffs,K,L2,npixels);
incode=startpos;
inweight=in.weight;
inmask=in.mask;
end

mask=(sum(inmask==1));
outweight=inweight;
outcode=incode;
hscslice=zeros(size(incode));

for sup=unique(mask)
	if sup>1
		I=find(mask==sup);
                II=find(inmask(:,I(1))==1);
                %slice=in.coeffs(II,I);
                slice=coeffs(II,I,:);
                wslice=inweight(II,I);
                cslice=incode(II,I);
		[cout,wout,posout,hscout]=haar_weight(slice,wslice,cslice);
                cout_tmp=reshape(cout,size(cout,1),numel(cout)/size(cout,1));
                cout_tmp=flipud(cout_tmp);
                cout=reshape(cout_tmp,size(cout));
                coeffs(II,I,:)=cout;
                outweight(II,I)=flipud(wout);
                outcode(II,I)=flipud(posout);
                hscslice(II,I)=flipud(hscout);
	end
end

out.weight=outweight(:,1:L2);
out.coeffs=reshape(coeffs,size(out.coeffs));

%encode haar transform
if dim < m
%transform along orientations
startpos=mod(floor(out.code_or_pos/L^dim),L);
out.code_or_pos = out.code_or_pos - L^dim*(mod(floor(out.code_or_pos/L^dim),L));
out.code_or_pos = out.code_or_pos + L^dim*outcode(:,1:L2);
out.code_or_hsc = out.code_or_hsc + L^dim*hscslice(:,1:L2);
else
%transform along scales
startpos=mod(floor(out.code_sc_pos/J^(dim-m)),J);
out.code_sc_pos = out.code_sc_pos - J^(dim-m)*(mod(floor(out.code_sc_pos/J^(dim-m)),J));
out.code_sc_pos = out.code_sc_pos + J^(dim-m)*outcode(:,1:L2);
out.code_sc_hsc = out.code_sc_hsc + J^(dim-m)*hscslice(:,1:L2);
end



end


function [cout,wout,outpos, outscale]=haar_weight(in,weights,positions)

[N,L,X]=size(in);
J=ceil(log2(N));


outpos=positions(:,1);

discrep=positions-outpos*ones(1,L);
if norm(discrep(:))>0
 error('this should not happen')
end
outscale=zeros(N,1);

cout=in;
wout=weights;
start=1;
for j=1:J
  [cout(start:end,:,:),wout(start:end,:),outpos(start:end),outscale(start:end)]=...
    haar_atomic(cout(start:end,:,:),wout(start:end,:),outpos(start:end),outscale(start:end));
  start=start+floor((N-start+1)/2);
end

outpos=outpos*ones(1,L);
outscale=outscale*ones(1,L);

end


function [out,wout,outpos,outscale]=haar_atomic(in,win,inpos,inscale)

%basic decomposition step into average and difference
[N,M,X]=size(in);
out=in;
wout=win;
K=2*floor(N/2);
outpos=inpos;
outscale=inscale;

  W1=win(1:2:K,:);
  W2=win(2:2:K,:);
  S1=in(1:2:K,:,:);
  S2=in(2:2:K,:,:);
  W=sqrt(W1.^2+W2.^2);
  W1b=reshape(repmat(W1,[1 X]),[size(W1) X]);
  W2b=reshape(repmat(W2,[1 X]),[size(W2) X]);
  Wb=reshape(repmat(W,[1 X]),[size(W) X]);
  M=(W1b.*S1+W2b.*S2)./Wb;
  D=(W2b.*S1-W1b.*S2)./Wb;
  out(1:K/2,:,:)=D;
  out(K/2+1:K,:,:)=M;
  wout(1:K/2,:)=W;
  wout(K/2+1:K,:)=W;
  
  pos1=inpos(1:2:K);
  pos2=inpos(2:2:K);
  sc1=inscale(1:2:K);
  sc2=inscale(2:2:K);
  outpos(1:K/2)=min(pos1,pos2);
  outpos(K/2+1:K)=min(pos1,pos2);
  outscale(1:K/2)=max(sc1,sc2)+0;
  outscale(K/2+1:K)=max(sc1,sc2)+1;  

end


