% function C=confusionmatrix(labels,Ncat)
% % each line = classif results for cat i
% Nsamplepercat=numel(labels)/Ncat;
% 
% C=zeros(Ncat,Ncat);
% for k=1:Ncat
%    h=hist([0,labels((k-1)*Nsamplepercat+(1:Nsamplepercat)),Ncat+1],Ncat+2) ;
%    h=h(2:end-1);
%    C(k,:)=h;
% end
% %C=C/Nsamplepercat;
% end
function C=confusionmatrix(labels,truelabels)
% each line = classif results for cat i
Ncat=max(truelabels);

C=zeros(Ncat,Ncat);
for k=1:Ncat
   
   labelsk=labels(truelabels==k);
    
   h=hist([0;labelsk;Ncat+1],Ncat+2) ;
   h=h(2:end-1);
   C(k,:)=h;
end
%C=C/Nsamplepercat;
end