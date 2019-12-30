function draw_scat_slices(transf,meta)


		%mode=0 : group by orientation
		%mode=1 : group by scale
		%mode=2 : display both groupings

		mode=1;

		if ndims(transf)>2|~isvector(transf)
				fprintf('ho')
				st=size(transf);
				transf=reshape(transf,numel(transf)/st(end),st(end));
				transf=mean(transf.^2);
		end

		factor=30000;

		differ=0;
		first_mask=find(meta.order==2);
		J=max(meta.scale(first_mask))+1;
		L=max(meta.orientation(first_mask))+1;
		figure
		title('first order coeffs')
		template=zeros(J,L);
		for j=1:J
				for l=1:L
						supo=find((meta.orientation== l-1).*(meta.scale==j-1).*(meta.order==2));
						template(j,l)=transf(supo);
				end
		end
		imagesc(template)
		colormap gray



		if mode==2 | mode==0 
				figure
				title('scale plots')
				for l1=1:L
						for l2=1:L
								supo=find((meta.orientation == (l1-1) + L*(l2-1)).*(meta.order==3));
								template=zeros(J);
								for rast=supo
										%tmp=transf(:,rast);
										if differ
												code_ori = myind2sub(meta.scale(rast),J,2);
												code_shifted=zeros(1,2);
												code_shifted(1) = code_ori(2);
												code_combined = mod(code_ori - code_shifted,J);
												code_sc = mysub2ind(code_combined,J,2);
										else
												code_sc=meta.scale(rast);
										end

										template(1+mod(code_sc,J),1+mod(floor(code_sc/J),J))=transf(rast);%mean(tmp(:));
										%template(1+mod(meta.scale(rast),J),1+mod(floor(meta.scale(rast)/J),J))=mean(tmp(:));
								end
								subplot(L,L,l1+L*(l2-1))
								image(factor*template)
								%imagesc(template)
						end
				end
				colormap gray
		end

		if mode==1|mode==2
				figure
				title('orientation plots')
				for j1=1:J
						for j2=1:J
								supo=find((meta.scale == (j1-1) + J*(j2-1)).*(meta.order==3));
								template=zeros(L);
								for rast=supo
										%tmp=transf(:,rast);
										template(1+mod(meta.orientation(rast),L),1+mod(floor(meta.orientation(rast)/L),L))=transf(rast);%mean(tmp(:));
								end
								subplot(J,J,j1+J*(j2-1))
								image(factor*template)
								%imagesc(template)
						end
				end
				colormap gray
		end


