function display_mnist_digits(in,options)


		options.aa=1;
		%scatt transform
		[out,met]=scatt(in,options);


		figure
		S=size(out);
		outslice=reshape(out,S(1)*S(2),S(3));
		outmaxs=max(outslice);
		size(outmaxs)
		ratios=outmaxs./met.dirac_norm;
		factor=64/max(ratios);

		for s1=2:2:S(1)
				for s2=2:2:S(2)
						paint=fulldisplay2d(squeeze(out(s1,s2,:))',met,0,2,1);
						subplot(S(1)/2,S(2)/2,s1/2+(s2/2-1)*S(1)/2)
						image(factor*paint{1})
						axis off
				end
		end


