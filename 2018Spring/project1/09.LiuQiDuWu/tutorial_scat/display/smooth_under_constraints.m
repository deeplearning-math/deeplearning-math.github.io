function [out,out0,basis,A,B]=smooth_under_constraints(intervals, energies)
%proof of concept

%intervals contains an array of right corners
%intervals is of size Kx2. k-th row contains (Ik_inf, Ik_sup) 
N=max(intervals(:));
%they should be disjoint.

if isvector(intervals)
start=ones(size(intervals));
start(2:end)=intervals(1:end-1)+1;
intervals=[start intervals]; 
end

[K,res]=size(intervals)

intlengths=intervals(:,2)-intervals(:,1)+1;

%construct the orthonormal basis 
basis=[];
lowp=[];
group=[];
lgroup=[];
g0=zeros(1,N);
for k=1:K
	support=intervals(k,1):intervals(k,2);

	if intlengths(k)>1
	chunk=haarbasis(intlengths(k));
	newvectors=zeros(N,size(chunk,2));
	newvectors(support,:)=chunk;
	basis=[basis newvectors];
	group=[group k*ones(1,size(chunk,2))];
	end
	
	lowpass=zeros(N,1);
	lowpass(support)=1/sqrt(intlengths(k));
	lowp=[lowp lowpass];
	lgroup=[lgroup k];
	g0(support)=energies(k)/intlengths(k);
	
end

correction=[-N/2+1:N/2];
correction=fftshift(abs(correction).^1);
fbasis=fft(basis).*(correction'*ones(1,size(basis,2)));
f0=fft(g0).*correction;

[N,L]=size(basis);

B=-.5*(f0*conj(fbasis) + conj(f0)*fbasis);
A=real(fbasis'*fbasis);
betas=linsolve(A,transpose(B));

out=g0'+basis*betas;


%init point is sqrt(abs(out));

out0=sqrt(abs(out));

basis=[basis lowp];
group=[group lgroup];
fbasis=fft(basis).*(correction'*ones(1,size(basis,2)));
A=(fbasis'*fbasis);

%gradient descent with alternate projections
betas=out0'*basis;
betas=betas';
maxiters=801;
step=-0.01; %?
for it=1:maxiters

	gradi=real(A)*betas;

	if norm(gradi) > 1
	gradi=gradi/norm(gradi);
	end
	
	betas=betas+step*norm(betas)*gradi;
	
	%reprojection
	for k=1:K
		supo=find(group==k);
		betas(supo)=betas(supo)*sqrt(energies(k)/sum(betas(supo).^2));
	end

end

out=basis*betas;

end


function chunk=haarbasis(L)

	if L>1
		n=round(L/2);
		m=L-n;
		c1=haarbasis(n);
		c2=haarbasis(m);
		chunk=zeros(L,1);
		chunk(1:n)=1/sqrt(n+n^2/m);
		chunk(n+1:L)=-chunk(1)*n/m;
		if ~isempty(c1)
			d1=zeros(L,size(c1,2));
			d1(1:n,:)=c1;
			chunk=[chunk d1];
		end
		if ~isempty(c2)
			d1=zeros(L,size(c2,2));
			d1(n+1:L,:)=c2;
			chunk=[chunk d1];
		end
	else
		chunk=[];
	end

end



