fileFolder1=fullfile('SAMPLE/train/good_0/');
dirOutput1=dir(fullfile(fileFolder1,'*.jpg'));
fileNames_good={dirOutput1.name}';

fileFolder2=fullfile('SAMPLE/train/bad_1/');
dirOutput2=dir(fullfile(fileFolder2,'*.jpg'));
fileNames_bad={dirOutput2.name}';
di=224;
x_good=zeros(di,di,27000,'uint8');
x_bad=zeros(di,di,3000,'uint8');
for i=1:27000
   x_good(:,:,i)=imread(['SAMPLE/train/good_0/',fileNames_good{i,1}]);
end

for i=1:3000
    x_bad(:,:,i)=imread(['SAMPLE/train/bad_1/',fileNames_bad{i,1}]);
end

Nx_good=im2double(x_good);
Nx_bad=im2double(x_bad);

img=zeros(di,di,30000);
for i=1:27000
img(:,:,i)=Nx_good(:,:,i);
end
for i=27001:30000
img(:,:,i)=Nx_bad(:,:,i-27000);
end
  savePath ='SAMPLE/IMG.mat';
   save(savePath,'img','-v7.3');
   label=[zeros(27000,1);ones(3000,1)];
   csvwrite('SAMPLE/label.csv',label);
   
    a=randperm(30000);
    img_r=img(:,:,a);
    label_r=label(a,1);
    savePath ='SAMPLE/IMG_R.mat';
   save(savePath,'img_r','-v7.3');
   csvwrite('SAMPLE/label_r.csv',label_r);
    
    