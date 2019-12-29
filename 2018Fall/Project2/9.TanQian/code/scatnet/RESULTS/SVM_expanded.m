
feac=csvread('feature_clear.csv');
feat=csvread('feature_test.csv');
unknown_fea=csvread('feature_unknown.csv');

labc=csvread('label_clear.csv');
labt=csvread('label_test.csv');
SVMModel=fitcsvm(feac,labc);
num1=size(labc,1);num2=size(labt,1);
labeltrain= predict(SVMModel,feac);
labeltest = predict(SVMModel,feat);
label_unknown = predict(SVMModel,unknown_fea);
csvwrite('label_svm_expanded_pre/label_unknown.csv',label_unknown);

training_accuracy=sum(abs(labeltrain-labc)==0)/num1;
testing_accuracy=sum(abs(labeltest-labt)==0)/num2;

bad_accuracy=sum(labt.*labeltest)/sum(labt);
good_accuracy=sum((labt==0).*(labeltest==0))/sum(labt==0);
fprintf('%f\n',training_accuracy);
fprintf('%f\n',testing_accuracy);
fprintf('%f\n',good_accuracy);
fprintf('%f\n',bad_accuracy);