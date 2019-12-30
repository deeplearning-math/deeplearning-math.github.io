function [out,l1norms,l1orig,l1eq]=equalize_first_order_scattering(f,g,options,psi,phi,lp)
	%this function equalizes the process g such that
	%it has the same first order scattering coefficients as f
	
	tight = getoptions(options,'tight_frame',1);

	J=size(psi,2);
	L=size(psi{end},2);

	% calculate dual wavelets
	[dpsi,dphi]=dualwavelets(psi,phi,lp{1},tight);

	%cubic spline is a tight frame
	%dpsi=psi;
	%dphi=phi;

	% calculate wavelet transform of original (f) and current (g)
	[FW,FPhi]=wavelet_fwd(f,psi,phi,options);
	[GW,GPhi]=wavelet_fwd(g,psi,phi,options);

	% average wavelet modulus coefficients to get the expected scattering
	for j=1:J
		for l=1:L
			l1norms(j,l)=sum(abs(FW{j}{l}(:)));
			l1orig(j,l)=sum(abs(GW{j}{l}(:)));
		end
	end

	% l1 norm of original process, never used
	l1f=sum(abs(f(:)));

	niters=16;

	for n=1:niters
		% equalize GW
		GW = equalize(GW,l1norms,J,L);
		% reproducing kernel
		[GW,GPhi,out]=wavelet_rk(GW,GPhi,psi,phi,dpsi,dphi,options);
	end

	% how close did we end up?
	for j=1:J
		for l=1:L
			l1eq(j,l)=sum(abs(GW{j}{l}(:)));
		end
	end

end

function Wout=equalize(W,l1norms,J,L)
	tol=1e-10;
	for j=1:J
		for l=1:L
			% what is current expected scattering
			temp=sum(abs(W{j}{l}(:)));
			% don't equalize if too small (unstable)
			if temp>tol
				Wout{j}{l}=(l1norms(j,l)/temp)*W{j}{l};
			else
				Wout{j}{l}=W{j}{l};
			end
		end
	end
end





