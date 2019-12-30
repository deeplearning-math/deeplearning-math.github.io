function out=mysub2ind(code,base,len)

out=0;
for l=len:-1:1
out=base*out+code(l);
end

end


