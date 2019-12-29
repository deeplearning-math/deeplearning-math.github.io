fea=csvread('2_feature.csv');
lab=csvread('label.csv');
figure(1);set(figure(1),'visible','off');
%gscatter(fea(:,1),fea(:,2),lab(1:end,:));
% n=4;
% for i=1:n
%     for j=1:n
%     subplot(n,n,(i-1)*n+j);
%     gscatter(fea(:,i),fea(:,j),lab(1:end,:));
%     legend('off');
%     end
%  end
for i=1:size(fea,1)
    if lab(i,:)==0
        plot(fea(i,1),fea(i,2),'r.');
        hold on;
    end
    if lab(i,:)==1
        plot(fea(i,1),fea(i,2),'b.');
        hold on;
    end
 
end
legend('good','bad');


imgname=['visualization.png'];
print(gcf,'-dpng',imgname);