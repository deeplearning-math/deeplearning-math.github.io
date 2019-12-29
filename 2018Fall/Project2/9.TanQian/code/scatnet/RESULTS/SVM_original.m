
fea=csvread('feature.csv');
unknown_fea=csvread('feature_unknown.csv');
lab=csvread('label.csv');
feac=fea(1:25000,:);
feat=fea(25001:end,:);
labc=lab(1:25000,:);
labt=lab(25001:end,:);
SVMModel=fitcsvm(feac,labc);
num1=size(labc,1);num2=size(labt,1);
labeltrain= predict(SVMModel,feac);
labeltest = predict(SVMModel,feat);
label_unknown = predict(SVMModel,unknown_fea);
csvwrite('label_svm_original_pre/label_unk nown.csv',label_unknown);

training_accuracy=sum(abs(labeltrain-labc)==0)/num1;
testing_accuracy=sum(abs(labeltest-labt)==0)/num2;
bad_accuracy=sum(labt.*labeltest)/sum(labt);
good_accuracy=sum((labt==0).*(labeltest==0))/sum(labt==0);
fprintf('%f\n',training_accuracy);
fprintf('%f\n',testing_accuracy);
fprintf('%f\n',good_accuracy);
fprintf('%f\n',bad_accuracy);