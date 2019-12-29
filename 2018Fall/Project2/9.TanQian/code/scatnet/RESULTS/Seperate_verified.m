fea=csvread('feature.csv');
lab=csvread('label.csv');
num1=25000;num2=5000;
feac=zeros(num1,size(fea,2));
labc=zeros(num1,1);
feat=zeros(num2,size(fea,2));
labt=zeros(num2,1);

feac(:,:)=fea(1:num1,:);
labc(:,1)=lab(1:num1,1);
      
repnum=9;
feac_expd=zeros(num1+(repnum-1)*sum(labc),size(fea,2));
labc_expd=zeros(num1+(repnum-1)*sum(labc),1);
flag=find(labc==1);
feac_expd(1:num1,:)=feac(:,:);
labc_expd(1:num1,1)=labc(:,:);
feac_expd(num1+1:num1+(repnum-1)*sum(labc),:)=feac(repmat(flag,(repnum-1),1),:);
labc_expd(num1+1:num1+(repnum-1)*sum(labc))=1;


a=randperm(num1+(repnum-1)*sum(labc));
feac_expd_rand=feac_expd(a,:);
labc_expd_rand=labc_expd(a,1);



feat(:,:)=fea(num1+1:num1+num2,:);
labt(:,1)=lab(num1+1:num1+num2,1);


csvwrite('feature_clear.csv',feac_expd_rand);
csvwrite('label_clear.csv',labc_expd_rand);
csvwrite('feature_test.csv',feat);
csvwrite('label_test.csv',labt);
% SVMModel=fitcsvm(feac,labc);
% labeltest = predict(SVMModel,feat);