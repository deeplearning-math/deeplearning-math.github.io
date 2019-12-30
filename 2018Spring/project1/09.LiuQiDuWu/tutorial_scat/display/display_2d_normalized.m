function [out_orientation,out_scale,mu_orientation,mu_scale]=display_2d_normalized(meta_signal,meta_dirac)

%this function displays the second order scattering coefficients
%I need to use:
%meta.norm
%meta.order
%meta.scale
%meta.orientation

normalize=1;

target_size=512;
first_mask=find(meta_dirac.order==2);
J=max(meta_dirac.scale(first_mask))+1;
L=max(meta_dirac.orientation(first_mask))+1;


%First display: orientation plane

%compute the measures mu(theta1,theta2) from meta_dirac.norm

mu_orientation=zeros(L);
out_orientation=zeros(L);
for l1=1:L
for l2=1:L
	I=find((meta_dirac.orientation==(l2-1)+L*(l1-1))&(meta_dirac.order==3));
	mu_orientation(l1,l2)=sum(meta_dirac.norm(I).^2);
	if normalize
	out_orientation(l1,l2)=sum(meta_signal.norm(I))/sum(meta_dirac.norm(I));
	else
	out_orientation(l1,l2)=sum(meta_signal.norm(I));
	end
end
end

display_n(out_orientation,mu_orientation,target_size);

%Second display: scale plane
mu_scale=zeros(J-1);
out_scale=zeros(J-1);
for l1=1:J-1
for l2=1:J-1
	I=find((meta_dirac.scale==(l2+l1-1)+J*(l1-1))&(meta_dirac.order==3));
	mu_scale(l1,l2)=sum(meta_dirac.norm(I).^2);
	if normalize
	out_scale(l1,l2)=sum(meta_signal.norm(I))/sum(meta_dirac.norm(I));
	else
	out_scale(l1,l2)=sum(meta_signal.norm(I));
	end
end
end

display_n(out_scale,mu_scale,target_size);


end


function out=display_n(signal,mu,outsize)

[N,M]=size(mu);

marginals=sum(mu,2);

rn=1;
for n=1:N
	iin=rn:rn+round(outsize*marginals(n)/sum(marginals));
	rm=1;
	for m=1:M
		iim=rm:rm+round(outsize*mu(n,m)/marginals(n));
		out(iin,iim)=signal(n,m);
		rm=rm+round(outsize*mu(n,m)/marginals(n))+1;
	end
	rn=rn+round(outsize*marginals(n)/sum(marginals))+1;	
end


figure
imagesc(out)
colormap gray

end
