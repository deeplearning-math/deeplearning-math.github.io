 x = csvread("d:/x.csv");
 y = csvread("d:/y.csv")';
 uG=mean(x,1);%global mean of features
 uc=zeros(10,3969);%class mean of features
 for i=1:10
    s=find(y==i-1);
    S=x(s,:);
    uc(i ,:)=mean(S,1);
 end
 UG=uG(ones(10000,1),:);%copy 10000 times
 sigT=((x-UG)'*(x-UG))/10000;%total cov
 UG2=uG(ones(10,1),:);%copy 10 times
 sigB=(uc-UG2)'*(uc-UG2)/10;% between class cov
 UC=uc(ones(10000,1),:);%copy 10000 times
 sigW=((x-UC)'*(x-UC))/10000;%with class cov
 error1=max(max(sigT-(sigB+sigW)))/max(max(sigT)); %closeness of sigT and sigB+sigW
 
 temp1=trace(sigW*pinv(sigB))/10;
 Temp_matrix=uc-UG2;% 10*3969
 t=zeros(1,10);
 for i=1:10
     t(i)=norm(Temp_matrix(i,:),2);
 end
 temp2=std(t)/mean(t);
 error2=(temp1-temp2)/temp1;%closeness of 1st comparison

 
 for i=1:10
    Temp_matrix(i,:)=Temp_matrix(i,:)/norm(Temp_matrix(i,:),2);
 end
 Temp_matrix2=abs(Temp_matrix+1/9);
 temp3=sum(sum(Temp_matrix2))/100;
 Temp_matrix3=triu(Temp_matrix);
 Temp_matrix4=Temp_matrix3(find(Temp_matrix3~=0));
 Temp_matrix5=[Temp_matrix4;Temp_matrix4];
 temp4=std(Temp_matrix5);
 error3=(temp3-temp4)/temp3;
 
 