function [train,test] = retrieve_ou_tex_10(Ltrain, Ltest, maxclasses)
% training contains 24 (class) * 20 (samples ) = 480 or 20 per class
% testing contains 24 (class) * 20 (samples)* 8 (orientation) = 3840 or 160
% per class

startup;
%function [train, test] = retrieve_mnist_rotate_s(Ltrain, Ltest, maxclasses)
train_list=fullfile(mpath,'Outex_TC_00010','000','train.txt');
fid=fopen(train_list);
a=textscan(fid,'%s %d');
train_name=a{1}(2:end);
train_lab=a{2}(2:end);
fclose(fid);
test_list=fullfile(mpath,'Outex_TC_00010','000','test.txt');
fid=fopen(test_list);
a=textscan(fid,'%s %d');
test_name=a{1}(2:end);
test_lab=a{2}(2:end);
fclose(fid);

for i=1:maxclasses
    ind=find(train_lab==i-1);
    for j=1:Ltrain
        
        im_name=train_name(ind(j));
        im_name_full=fullfile(mpath,'Outex_TC_00010','images',im_name{1});
        train{i}{j}=imreadBW(im_name_full);
    end
    
    ind=find(test_lab==i-1);
    for j=1:Ltest
        
        im_name=test_name(ind(j));
        im_name_full=fullfile(mpath,'Outex_TC_00010','images',im_name{1});
        test{i}{j}=imreadBW(im_name_full);
    end
end


%label goes from 0 to 23

%end