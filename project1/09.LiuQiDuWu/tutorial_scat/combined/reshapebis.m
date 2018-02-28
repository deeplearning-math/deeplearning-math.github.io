function b=reshapebis(A,n,m)
%reshape by put line 1 then line 2 then line 3...
%inverse of matlab builtin
b=reshape(A',m,n)';
