function [training_label_vector,training_instance_matrix,testing_label_vector, ...
    testing_instance_matrix,training_indice_vector,testing_indice_vector]=classif_formatForSvmLibWithDefinedTraining(feat,Ntrain)

d=numel(feat{1}{1});
nbclass=numel(feat);

%randomly chose training and testing samples
for i=1:nbclass
    nbsample=numel(feat{i});
    trainingInd=1:nbsample;
    testingIndSt{i}=trainingInd(Ntrain+1:end);
    trainingIndSt{i}=trainingInd(1:Ntrain);
end


%fill training and testing matrix with data
training_label_vector=zeros(nbclass*Ntrain,1);
training_indice_vector=zeros(nbclass*Ntrain,1);
training_instance_matrix=zeros(nbclass*Ntrain,d);
for i = 1:nbclass
    training_label_vector((i-1)*Ntrain + (1:Ntrain)) = i;
    for j = 1:Ntrain
        training_indice_vector((i-1)*Ntrain + j)=trainingIndSt{i}(j);
        training_instance_matrix((i-1)*Ntrain + j,:)=reshape(feat{i}{trainingIndSt{i}(j)},1,d);
    end
end

nbtest=0;
for i = 1:nbclass
    nbtest=nbtest+numel(testingIndSt{i});
end

testing_label_vector=zeros(nbtest,1);
testing_indice_vector=zeros(nbtest,1);
testing_instance_matrix=zeros(nbtest,d);
curj=1;
for i = 1:nbclass
    for j = 1:numel(testingIndSt{i})
        testing_label_vector(curj) = i;
        testing_indice_vector(curj)=testingIndSt{i}(j);
        testing_instance_matrix(curj,:) = reshape(feat{i}{testingIndSt{i}(j)},1,d);
        curj = curj+1;
    end
end

