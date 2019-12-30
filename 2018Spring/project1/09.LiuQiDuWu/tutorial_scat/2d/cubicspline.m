function filts=cubicspline(N,options)

	[h0,g0,h,g]=Splinewavelet(3,N);
	
	J=getoptions(options,'J',4);
	filts.psi{1}=fft(g0)/sqrt(2);
	for j=2:J
		filts.psi{j}=zeros(size(filts.psi{j-1}));
		slice=filts.psi{j-1}(1:2:end);
		filts.psi{j}(1:length(slice))=slice;
	end
	filts.phi = zeros(size(filts.psi{1}));
	slice=fft(h0)/sqrt(2);
	L=length(slice);
	slice=slice(1:L/2);
	slice=slice(1:2^(J-1):end);
	L=length(slice);
	filts.phi(1:L)=slice;
	%filts.phi(end:-1:end-L+2)=slice(2:end);

	filts.littlewood=zeros(size(filts.psi{1}));

	filts.littlewood = filts.littlewood + abs(filts.phi).^2;
	for j=1:J
		filts.littlewood = filts.littlewood + .5* abs(filts.psi{j}).^2;
	end
