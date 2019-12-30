
function out=myind2sub(code,base,len)

L=length(code);
out=zeros(L,len);

for l=1:len
	out(:,l)=mod(floor(code/(base^(l-1))),base);
end

end


