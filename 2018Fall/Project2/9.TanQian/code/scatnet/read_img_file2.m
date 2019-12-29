fileFolder1=fullfile('SAMPLE/test/all_tests/');
dirOutput1=dir(fullfile(fileFolder1,'*.jpg'));
fileNames_test={dirOutput1.name}';
di=224;
x_test=zeros(di,di,3000,'uint8');

for i=1:3000
   x_test(:,:,i)=imread(['SAMPLE/test/all_tests/',fileNames_test{i,1}]);
end



Nx_test=im2double(x_test);



  savePath ='SAMPLE/IMG_unknown.mat';
   save(savePath,'Nx_test','-v7.3');

    
    