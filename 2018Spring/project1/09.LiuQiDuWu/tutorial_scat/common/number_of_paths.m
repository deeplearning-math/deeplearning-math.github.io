function out=number_of_paths(in)
%NUMBER_OF_PATHS Calculate the number of paths in a 'raw' scattering object
	out=0;
	S=size(in);

	for s=1:S(2)
		out=out+size(in{s},2);
	end
end

