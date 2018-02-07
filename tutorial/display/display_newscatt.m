function [bigim] = display_newscatt(sf,options)
% options are
% todisp = [SJ] / S
options.null=1;
celltodisp_name=getoptions(options,'todisp','SJ');
celltodisp=eval(['sf.',celltodisp_name]);
[N M]=size(celltodisp{1}{1}.signal);
margin=getoptions(options,'margin',10);
[bounds]=getoptions(options,'bounds',[1,N,1,M]);
Nmin=bounds(1);
Nmax=bounds(2);
Mmin=bounds(3);
Mmax=bounds(4);
J=0;
L=0;
for i=1:numel(celltodisp{2})
    J=max(J,celltodisp{2}{i}.meta.scale);
    L=max(L,celltodisp{2}{i}.meta.orientation);
end
J=J+1;
L=L+1;

bigN=J*(Nmax-Nmin+1) + margin*(J-1);
bigM=L*(Mmax-Mmin+1) + margin*(L-1);
bigim=zeros(bigN,bigM);
for j=1:J
    for th=1:L
        sig=celltodisp{2}{(j-1)*L + th}.signal;
        [n m]=size(sig);
        if (n==N)&&(m==M)
            sig=sig((Nmin:Nmax),(Mmin:Mmax));
            bigim( (1:(Nmax-Nmin+1)) + (Nmax-Nmin+1+margin)*(j-1) , (1:(Mmax-Mmin+1)) + (Mmax-Mmin+1+margin)*(th-1) ) = sig;
        else
            bigim( (1:n) + (N+margin)*(j-1) , (1:m) + (M+margin)*(th-1) ) = sig;
        end
    end
end

imagesc(bigim);

