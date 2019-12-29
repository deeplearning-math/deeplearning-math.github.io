function [imgs,labs]=read_raf(num)
f1=fopen('./RAF/Raphael_Project_final/Raphael_Project_final/name.txt','r');
imgP=zeros(8000,8000,3*num,'uint8');
siz=zeros(num,3);
for i=1:num
    si=fgetl(f1);
    fname=sprintf('./RAF/Raphael_Project_final/Raphael_Project_final/%s',si);
    a=imread(fname);
    siz(i,:)=size(a);
    fprintf('%d %d %d %d\n',i,siz(i,:));
    imgP(1:siz(i,1),1:siz(i,2),((i-1)*3+1):(i*3))=a(:,:,1:3);
end

m_=min(min(siz(:,1:2)));
imgs=zeros(m_,m_,3*num,'uint8');
for i=1:num
    fprintf('%d\n',i);
   % fprintf('%d %d %d \n',fix((siz(i,1)-m_)/2),fix((siz(i,1)-m_)/2+m_),fix((siz(i,1)-m_)/2+m_-1)-fix((siz(i,1)-m_)/2));
    % fprintf('%d %d %d \n',fix((siz(i,2)-m_)/2),fix((siz(i,2)-m_)/2+m_),fix((siz(i,2)-m_)/2+m_-1)-fix((siz(i,2)-m_)/2));
    imgs(:,:,((i-1)*3+1):(i*3))=imgP(fix((siz(i,1)-m_)/2+1):fix((siz(i,1)-m_)/2+m_),fix((siz(i,2)-m_)/2+1):fix((siz(i,2)-m_)/2+m_),1:3);
end
fclose(f1);
labs=load('./RAF/la.txt');
end
