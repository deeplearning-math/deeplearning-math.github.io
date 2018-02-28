function [imout,orderout,meta]=fulldisplay1d(scatt,meta,options)


		%type=0 unified plot
		%type=2 split plot
		%type=1 concatenated plot

		type=getoptions(options,'display_type',2);
		renorm_process=getoptions(options,'renorm_process',0);
		use_average=getoptions(options,'use_average',0);

		maxsize=2048;
		first_mask=find(meta.order==2);
		J=max(meta.scale(first_mask))+1;

		meta.covered=zeros(size(meta.order));
		meta=effective_energy(meta);
		meta=compute_rectangles(meta,type==0);
		maxorder=max(meta.order);

		if renorm_process
			if use_average
			norm_ratio = scatt./meta.dirac_ave;
			else
			norm_ratio = scatt./meta.dirac_norm;
			end
		else
			norm_ratio = scatt;
		end

		heights=meta.rectangle(:,2)-meta.rectangle(:,1);

		fact_h = maxsize;

		if type < 2
				imout=zeros(fact_h+1,1);
				orderout=zeros(fact_h+1,1);
		else
				for m=1:maxorder-1
						imout{m}=zeros(fact_h,1);
				end
		end

		if type==1
				%concatenate the rectangles along orders
				%rescale the horizontal coordinate according to total energy of each order
				for ord=1:maxorder
						selected=find(meta.order==ord);
						ordener(ord)=sum(meta.dirac_norm(selected).^2);
				end
				ordener=ordener/sum(ordener);
				cordener=cumsum(ordener);
				cordener=[0 cordener];
				%concatenate and squeeze the rectangles
				lower=0;
				for ord=1:maxorder
						selected=find(meta.order==ord);
						if ordener(ord)>0
								meta.rectangle(selected,1:2)=ordener(ord)*meta.rectangle(selected,1:2)+cordener(ord);
						end
				end
				type=0;
		end

		switch type
				case 0	
						for l=1:length(norm_ratio)
								%extrema
								ext(1)=1+floor(fact_h*meta.rectangle(l,1));
								ext(2)=1+floor(fact_h*meta.rectangle(l,2));

								inthh=[ext(1):ext(2)];
								%maskh=ones(size(inthh));
								%if length(maskh)>1
								%		maskh(1) = ceil(fact_h*meta.rectangle(l,1)) - (fact_h*meta.rectangle(l,1));
								%		maskh(end) = -floor(fact_h*meta.rectangle(l,2)) + (fact_h*meta.rectangle(l,2));
								%else
								%		maskh(1) = (fact_h*meta.rectangle(l,2))-(fact_h*meta.rectangle(l,1)); 
								%end

								%imout(inthh)=sqrt(imout(inthh).^2 + maskh*norm_ratio(l)^2);
								imout(inthh)=norm_ratio(l);
								orderout(inthh)=meta.order(l);

						end
						m1=(orderout==2);
						m2=(orderout==3);
						m3=(orderout>3);
						[gox]=gradient(orderout);
						couronnes=(conv(gox.^2,ones(5,1),'same') < .25);
						nimout=imout/max(imout(:));
						nimout=nimout.*couronnes+(1-couronnes);
						[NN,MM]=size(nimout);
						cimout=ones(NN,MM,3);
						cimout(:,:,1)=1-nimout.*(m2|m3);
						cimout(:,:,3)=1-nimout.*(m1|m3);
						cimout(:,:,2)=1-nimout.*(m1|m2);
						
				case 2
						for ord=2:maxorder
								selected=find(meta.order==ord);
								for l=selected
										inth=min(fact_h,[1+floor(fact_h*meta.rectangle(l,1)):floor(fact_h*meta.rectangle(l,2))]);
										imout{ord-1}(inth)=norm_ratio(l);
								end
						end		


				end

		end

function meta=compute_rectangles(meta,unified_plot)


		R=length(meta.order);
		maxorder=max(meta.order);
		first_mask=find(meta.order==2);
		J=max(meta.scale(first_mask))+1;

		%meta.lp_correction=min(1,meta.lp_correction);
		meta.rectangle(1,:)=[0 1];

		for o=1:maxorder-1
				slice=find(meta.order==o);
				for s=slice
						%find children
						if o==1
								children=find(meta.order==o+1);
						else
								children=find((floor(meta.scale/J)==meta.scale(s))&(meta.order==o+1));
						end
						if ~isempty(children)
								[newrectangles,outrect]=split_rectangle(meta.rectangle(s,:),meta.scale(children),...
										meta.dirac_norm(s),...
										meta.dirac_effnorms(children),J,unified_plot,o);
								for c=1:length(children)
										meta.rectangle(children(c),:)=newrectangles(c,:);
								end
								meta.covered(s) = ((outrect(2)-outrect(1))+...
										sum((newrectangles(:,2)-newrectangles(:,1))))/...
										((meta.rectangle(s,2)-meta.rectangle(s,1)));
								meta.rectangle(s,:)=outrect;
						end
				end
		end

end


function [out,outlowp]=split_rectangle(inrectangle, scales, dirac_phi, dirac_norms,J,unified_plot,order)

		%first step: we marginalize orientations in order to split scale axis:
		C=length(dirac_norms);
		if unified_plot | order==1
				ener=dirac_phi^2;
				totener=sum(dirac_norms)+ener;
		else
				totener=sum(dirac_norms);
		end
		sanity_checks=0;

		totalwidth=inrectangle(2)-inrectangle(1);

		outlowp=inrectangle;
		if unified_plot | order==1
				outlowp(end)=outlowp(end-1)+totalwidth*(ener/totener);
				rasterwidth=outlowp(end);
		else
				rasterwidth=inrectangle(1);
		end
		scale_parent= mod(floor(scales/J),J);

		for j=J-1:-1:0
				pack=find(mod(scales,J)==j);
				if ~isempty(pack)
						width=sum(dirac_norms(pack).^1);
						out(pack(1),1)=rasterwidth;
						out(pack(1),2)=rasterwidth+totalwidth*width/totener;
						rasterwidth=rasterwidth+totalwidth*width/totener;
				end
		end

		if order==3 & sanity_checks 
				%sanity check: conservation of energy		
				in_area = totalwidth;
				if unified_plot
					out_area = (outlowp(2)-outlowp(1)) + sum( (out(:,2)-out(:,1)));
				else
					out_area =  sum( (out(:,2)-out(:,1)));
				end

				tol=1e-5;
				if abs(in_area-out_area) > tol*in_area
						in_area
						out_area
						out
						inrectangle
						error('sthg weird')
				end
		end

end



function metaout=effective_energy(meta)

		R=length(meta.order);
		maxorder=max(meta.order);
		first_mask=find(meta.order==2);
		J=max(meta.scale(first_mask))+1;

		last_mask=find(meta.order==maxorder);
		metaout=meta;
		metaout.dirac_effnorms(last_mask)=meta.dirac_norm(last_mask).^2;

		for o=maxorder-1:-1:1
				slice=find(meta.order==o);
				for s=slice
						%find children
						if o==1
								children=find(meta.order==o+1);
						else
								children=find((floor(meta.scale/J)==meta.scale(s))&(meta.order==o+1));
						end
						metaout.dirac_effnorms(s)=sum(metaout.dirac_effnorms(children))+meta.dirac_norm(s).^2;
				end
		end

end


