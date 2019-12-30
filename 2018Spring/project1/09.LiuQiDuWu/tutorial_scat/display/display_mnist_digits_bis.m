function display_mnist_digits_bis(in,met,S,positive,moyenne)


		%options.aa=1;
		%scatt transform
		%[out,met]=scatt(in,options);

		outslice=reshape(in,S(1)*S(2),S(3));
                out=reshape(in,S(1),S(2),S(3));
if moyenne

clear out
out(2,2,:)=mean(outslice);
outslice=reshape(out,4,S(3));
S(1)=2;
S(2)=2;

end

marge=0.1;
cbar=0.3;
border=1;
lpos=(2-cbar)/S(2);
jpos=2/S(1);
lsize=lpos*(1-marge);
jsize=jpos*(1-marge);
loffs=lpos*marge*.5;
joffs=jpos*marge*.5;




		%S=size(out);
		%outmaxs=max(outslice)./met.dirac_norm;
		%outmins=min(outslice)./met.dirac_norm;
		outmaxs=max(abs(outslice))./met.dirac_norm;
		outmins=-outmaxs;
                palette=colormap(gray);
                if positive
                outmins=zeros(size(outmins));
                palette=1-colormap(gray);
                end
		masko1=find(met.order==2);
                masko2=find(met.order==3);
                offs1=min(outmins(masko1));
                offs2=min(outmins(masko2));
                vari1=max(outmaxs(masko1))-offs1; 
                vari2=max(outmaxs(masko2))-offs2; 

                sliceo2=outslice(:,masko2);
                fprintf('principal component added %f energy to second_order\n', sum(sliceo2(:))/sqrt(numel(sliceo2)))

		for s1=2:2:S(1)
				for s2=2:2:S(2)
						paint=fulldisplay2d(squeeze(out(s1,s2,:))',met,0,2,1);
                                                figure(1)
                                                subplot('position',[loffs+(s2/2-1)*lpos joffs+(S(1)/2-s1/2)*jpos loffs+lsize joffs+jsize]); 
						%subplot(S(1)/2,S(2)/2,s1/2+(s2/2-1)*S(1)/2)
						image(64*(paint{1}-offs1)/vari1)
						axis off
                                                figure(2)
                                                subplot('position',[loffs+(s2/2-1)*lpos joffs+(S(1)/2-s1/2)*jpos loffs+lsize joffs+jsize]); 
						%subplot(S(1)/2,S(2)/2,s1/2+(s2/2-1)*S(1)/2)
						image(64*(paint{2}-offs2)/vari2)
						axis off
				end
		end

          palette=colormap(jet); 
                figure(1)
               colormap(palette)  
                figure(2)
               colormap(palette) 

figure(1)
for t=1:6
label{t}=sprintf('%01.01f',offs1+(2*(t-1)+1)*vari1/12);
end
B=colorbar('YTickLabel',{label{1},label{2},label{3},label{4},label{5},label{6}}); 
set(B, 'Position', [.88 .11 .07 .8150]) 
set(B, 'FontSize',24)

figure(2)
for t=1:6
label{t}=sprintf('%01.01f',offs2+(2*(t-1)+1)*vari2/12);
end
B=colorbar('YTickLabel',{label{1},label{2},label{3},label{4},label{5},label{6}}); 
%B=colorbar; 
set(B, 'Position', [.88 .11 .07 .8150]) 
set(B, 'FontSize',24)

%for i1=1:2 
%for i2=1:2
%pos=get(A(i1,i2), 'Position'); 
%axes(A(i1,i2)) 
%set(A(i1,i2), 'Position', [pos(1) pos(2) .6626 pos(4)]) 
%end
%end
