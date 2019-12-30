function out =vectorize2(struct)
% special stuff to vectorize a struc along the columns for 1d haarcombscatt
n=0;
for m=1:size(struct,2)
    
    for p=1:size(struct{m},2)
        n=n+size(struct{m}{p},2);
    end
end
out=zeros(size(struct{1}{1},1),n);

n=1;
for m=1:size(struct,2) 
    for p=1:size(struct{m},2)
        out(:,n:n+size(struct{m}{p},2)-1)=struct{m}{p};
        n=n+size(struct{m}{p},2);
        
    end
end
