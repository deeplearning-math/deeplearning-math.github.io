function out=isprogressive(path,J,order)

code=myind2sub(path,J,order);
code2=zeros(size(code));
code2(1:end-1)=code(2:end);

out=(sum(code-code2<=0)==0);


