%%At first, you should download the scatnet-0.2 package
%%website:  http://www.di.ens.fr/data/software/scatnet/download/

foldername=dir('D:\download\deeplearning\scatnet-0.2\cropped_images');% 用于得出所有子文件夹的名字
new_list = [10,11,12,13,14,15,16,17,18,19,1,20,21,22,23,24,25,26,27,28,2,3,4,5,6,7,8,9]*200-200;
list = repmat([10,11,12,13,14,15,16,17,18,19,1,20,21,22,23,24,25,26,27,28,2,3,4,5,6,7,8,9]*200,200,1);
list = reshape(list,[1,5600]);
list1 = repmat([1:200]-200,1,28);
list2 = list +list1;

%%%process data
var1 =[];
select_list =[];
max_var=[];
select_number=100;
for i=1:28
    for j =200*i-199:200*i
        filename=strcat('D:\download\deeplearning\scatnet-0.2\cropped_images\',foldername(j+2).name);
        x=imreadBW(filename);
        var1(j)=var(x(:));
    end
end
for i=1:28
    [~,id]= sort(var1(200*i-199:200*i),'descend');
    select_list(i,:)=id(1:select_number)+new_list(i);
end
select_list = reshape(select_list,[1,select_number*28])
    
% sctenet to extract feature    
select_feature =[]   
cal =0
for i=[select_list]
    filename=strcat('D:\download\deeplearning\scatnet-0.2\cropped_images\',foldername(i+2).name);%读取子文件夹的名字和路径
    x=imreadBW(filename);% 读取图片
    %Wop = wavelet_factory_2d(size(x));
    % compute scattering with 5 scales, 6 orientations
    % and an oversampling factor of 2^2
    %S = scat(x, Wop);
% reformat in 3d matrix
    %S_mat = format_scat(S);
    Wop = wavelet_factory_2d(size(x));
    select_feature(i,:) =sum(sum(format_scat(scat(x,Wop)),2),3);
    %save('D:\deeplearning\scatnet-0.2',image1);
    cal=cal+1
    i
end
new_select_feature =[]
for i=[select_list]
    new_select_feature(list2(i),:) = select_feature(i,:);
    i;
end
 new_select_feature
new_select_feature(all(new_select_feature==0,2),:) = [];
save new_select_feature1.txt -ascii new_select_feature


