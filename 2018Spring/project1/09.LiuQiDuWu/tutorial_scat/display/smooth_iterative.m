function [g0,distance]=smooth_iterative(intervals, g0)


if isvector(intervals)
start=ones(size(intervals));
start(2:end)=intervals(1:end-1)+1;
intervals=[start intervals]; 
end

[K,res]=size(intervals);
maxiters=500;
sigma=2;
it=1;
distance=1;

if res==2 %1d signal
	intlengths=intervals(:,2)-intervals(:,1)+1;
	for k=1:K
		energies(k)=g0(intervals(k,1))^2*intlengths(k);
	end
elseif res==4 %2d signal
	intlengths=(intervals(:,2)-intervals(:,1)+1).*(intervals(:,4)-intervals(:,3)+1);
	lpf=fspecial('gaussian',ceil(16*sigma),sigma);
	for k=1:K
		energies(k)=g0(intervals(k,1),intervals(k,3))^2*intlengths(k);
	end
end

%g0=zeros(N,1);
%for k=1:K
%	support=intervals(k,1):intervals(k,2);
%	g0(support)=sqrt(energies(k)/intlengths(k));
%end

%gradient descent with alternate projections


while (it<maxiters)&(distance>2*1e-4)
	oldg0=g0;
	
	if(mod(it,16)==15)
			disp(it)
	end
	if res==2
		g0=gabor1dfilter(g0,sigma);
	elseif res==4
		g0=conv2(g0,lpf,'same');
	end

	%reprojection
	newg0=zeros(size(g0));
	for k=1:K
		if res==2
			support=intervals(k,1):intervals(k,2);
			if sum(g0(support).^2)
				newg0(support)=g0(support)*sqrt(energies(k)/sum(g0(support).^2));
			end
		elseif res==4
			supx=intervals(k,1):intervals(k,2);
			supy=intervals(k,3):intervals(k,4);
			if sum(sum(g0(supx,supy).^2))
				newg0(supx,supy)=g0(supx,supy)*sqrt(energies(k)/sum(sum(g0(supx,supy).^2)));
			end
		end
	end
	g0=newg0;
	distance=norm(g0(:)-oldg0(:))/norm(oldg0(:));
	it=it+1;
end

end

function out=gabor1d(sigma,N)

x=-N:N;

out=exp(-x.^2/(2*sigma^2));

out=out/norm(out);

end

function out=gabor1dfilter(in,sigma)

N=round(16*sigma);

g=gabor1d(sigma,N);
out=conv(in,g,'same');

end

