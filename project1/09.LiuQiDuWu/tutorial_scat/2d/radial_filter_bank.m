function [filters,dbgout] = radial_filter_bank (sizein, options)
		%function [filters] = radial_filter_bank (sizein, options)
		% This function implement a bank of radial filters for scattering decomposition
		% possible options[default] are : 
		%   J = the scales goes of wavelets go from a^0 to a^J [4]
		%   L = the number of orientation [6]
		%   a = the scale factor ( scale=a^j ) [2]
		%   radial_fun = a function handle for the radial_function at scale 1 [cauchy]
		%   angular_fun = a function handle for the angular function [gaussian]
		% See radial_filters_impl.pdf in the doc folder for detailed explanation
		options.null=1;

		J=getoptions(options,'J',4); % scale relative to a (meaning that maximum scale is a^(J-1)
		L=getoptions(options,'L',6);
		a=getoptions(options,'a',2);
		radial_fun = getoptions(options,'radial_fun',@(x) cauchy(x,options));
		options.gaussian_sigma=getoptions(options,'gaussian_sigma',0.5*4/L);
		angular_fun = getoptions(options,'angular_fun',@(x) gaussian(x,options));
		% the user specifies how many thetas between 0 and pi he wants but we need the
		% the ones between pi and 2*pi as well to compute the littlehood for angular.
		cubic=getoptions(options,'cubicspline',0);
		if cubic
		thetas=(0:1*L-1)  * pi / L;
		ss=9;
		sss=12;

		else
		thetas=(0:2*L-1)  * pi / L;
		end

		for res=0:floor(log2(a)*(J-1))

				N=ceil(sizein(1)/2^res);
				M=ceil(sizein(2)/2^res);
				%if cubic
				%	N=N/2;
				%	M=M/2;
				%end

				[omega2,omega1]=meshgrid(-M:M-1,-N:N-1);
				if cubic
					omega1=fftshift(omega1);
					omega2=fftshift(omega2);
				else
					omega1=fftshift(omega1)/N*2*pi;
					omega2=fftshift(omega2)/M*2*pi;
				end
				omega=1i*omega1+omega2;
				mod_omega=abs(omega);
				ang_omega=angle(omega);


				littlehood_rad=zeros(2*N,2*M);
				littlehood_ang=zeros(2*N,2*M);

				if cubic
						optis.J=J-res;
						splinesradiales=cubicspline(ss*N,optis);
						tempo=mod_omega(:);
				end

				for j=floor(res/log2(a)):J-1
						if cubic
								temp=splinesradiales.psi{j+1-floor(res/log2(a))}(min(ss*N,1+round(sss*tempo)));
								psif_rad{j+1}=reshape(temp,size(mod_omega));
						else
								scale=a^j*2^(-res);
								psif_rad{j+1} = radial_fun(scale*mod_omega);
								littlehood_rad = littlehood_rad + abs(psif_rad{j+1}.^2);
						end
				end


				for th=1:numel(thetas);
						if cubic
								phif_ang{th} = angularspline((ang_omega-(th-1)*pi/numel(thetas))*numel(thetas)/2);
						else
								phif_ang{th} = angular_fun( minMod(ang_omega-thetas(th),2*pi) ) ;
								littlehood_ang = littlehood_ang + abs(phif_ang{th}).^2;
						end
				end

				if cubic
						slice=splinesradiales.phi(min(ss*N,1+round(sss*tempo)));
						phif{res+1}=reshape(slice,size(mod_omega));
						tmp=real(ifft2(phif{res+1}));
						phif{res+1}=4*fft2(tmp(1:2:end,1:2:end));
						for j=floor(res/log2(a)):J-1
								for th=1:numel(thetas);
										%psif{res+1}{j+1}{th} = psif_rad{j+1}.*phif_ang{th};
										tmp = ifft2(psif_rad{j+1}.*phif_ang{th});
										psif{res+1}{j+1}{th}=4*fft2(tmp(1:2:end,1:2:end));
								end
						end
				else
						%find K_rad
						K_rad=max(max(littlehood_rad));

						%find R_cut
						%find every local max of littlheood
						local_max_littlehood_rad = ones(2*N,2*M);
						shifts= [ 1 1 ; 1 0 ; 1 -1; ; 0 1; 0 -1 ; -1 1; -1 0; -1 -1];
						for sh=1:size(shifts,1)
								local_max_littlehood_rad = local_max_littlehood_rad.*...
										(circshift(littlehood_rad,shifts(sh,:))<=littlehood_rad);
						end
						%find the one with smallest radius
						Rcut = min(mod_omega(find(local_max_littlehood_rad>0)));

						%find k_ang and K_ang
						k_ang=min(min(littlehood_ang));
						K_ang=max(max(littlehood_ang));

						%compute low pass filter phi
						mask = (mod_omega <= Rcut );
						tmpphi = ifft2( mask.*sqrt(1- littlehood_rad/K_rad));
						phif{res+1} = 4*fft2(tmpphi(1:2:end,1:2:end));

						%compute high pass filters psif{j}{theta}
						littlehood_final=zeros(N,M);
						littlehood_final=abs(phif{res+1}).^2;
						thetas=(0:L-1)  * pi / L;
						for j=floor(res/log2(a)):J-1
								for th=1:numel(thetas);
										tmp = ifft2(psif_rad{j+1}/sqrt(K_rad) .*...
												sqrt(2/(k_ang+K_ang)).*phif_ang{th});
										psif{res+1}{j+1}{th} = sqrt(2)*  4*fft2(tmp(1:2:end,1:2:end));% sqrt(2) since we keep only positive thetas, 4 is for downsampling
										littlehood_final = littlehood_final + abs(psif{res+1}{j+1}{th}).^2;
								end
						end
						if res==0
								dbgout=littlehood_final;
						end
				end

				filters.psi=psif;
				filters.phi=phif;
		end
		filters.psi=psif;
		filters.phi=phif;
		filters.a=a;

