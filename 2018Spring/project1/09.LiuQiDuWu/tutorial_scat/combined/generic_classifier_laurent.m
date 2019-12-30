function out=generic_classifier_laurent(feat_train,feat_test,classif_options)
classif=getoptions(classif_options,'classif','give me a classif method !');
randTrain=getoptions(classif_options,'randTrain',0);

switch randTrain
  case 0
    
    
    switch classif
      case 'kNN'
        split = getoptions(classif_options,'split',0);
        if (split==1)
          Ncut = getoptions(classif_options,'Ncut',20);
          %[feat_train,feat_test] = split_db(feat_train,Ncut);
          [training_label_vector,training_instance_matrix,...
            testing_label_vector,testing_instance_matrix,...
            training_indice_vector,testing_indice_vector]=...
            classif_formatForSvmLibWithDefinedTraining(feat_train,Ncut);
          [predicted_label,nearest_neighbors] = nnpredict(testing_label_vector,...
            testing_instance_matrix,training_label_vector,training_instance_matrix);
          out=classif_computeScore(predicted_label,testing_label_vector);
        else
          in=classif_distance(feat_train,feat_test,classif_options);
          for K=1:min(10,numel(feat_train{1}))
            out.detailed_results{K}=classif_kNN(in,K);
            out.score(K)=out.detailed_results{K}.score_avg;
          end
          if getoptions(classif_options,'storeDisMat',0)
            out.distanceMatrix=in.d;
          end
        end
        
      case 'PCA'
        out=classif_pcamodel_alld(feat_train,feat_test);
        
        
        
      otherwise
        error('unknown classif algorithm');
        
    end
  case 1
    Ntrain=getoptions(classif_options,'Ntrain',-1);
    if (Ntrain==-1)
      listOfNtrain=[5,10,20];
    else
      listOfNtrain=Ntrain;
    end
    out.listOfNtrain=listOfNtrain;
    switch classif
      
      case 'NNL2' %much simpler
        nRandTrain=getoptions(classif_options,'nRandTrain',16);
        
        
        for iNtrain=numel(listOfNtrain):-1:1
          
          
          Ntrain=listOfNtrain(iNtrain);
          for rt=1:nRandTrain
            %randomly split data between training
            [training_label_vector,training_instance_matrix,...
              testing_label_vector,testing_instance_matrix,...
              training_indice_vector,testing_indice_vector]=...
              classif_formatForSvmLibWithRandomTraining(feat_train,Ntrain);
            
            
            fprintf(' NN classif : Ntrain %d  rt %d  \n ',Ntrain,rt);
            [predicted_label,nearest_neighbors] = nnpredict(testing_label_vector,...
              testing_instance_matrix,training_label_vector,training_instance_matrix);
            
            outclassif=classif_computeScore(predicted_label,testing_label_vector);
            
            out.score_avg(iNtrain,rt)=outclassif.score_avg;
            out.score_per_cat(:,iNtrain,rt)=outclassif.score_per_cat;
            out.confusionMatrix(:,:,iNtrain,rt)=outclassif.confusionMatrix;
            out.training_indice_vector(1:numel(training_indice_vector),iNtrain,rt)=reshape(training_indice_vector,1,numel(training_indice_vector));
            out.testing_indice_vector(1:numel(testing_indice_vector),iNtrain,rt)=reshape(testing_indice_vector,1,numel(testing_indice_vector));
            out.predicted_label(1:numel(predicted_label),iNtrain,rt)=reshape(predicted_label,1,numel(predicted_label));
            out.true_label(1:numel(testing_label_vector),iNtrain,rt)=reshape(testing_label_vector,1,numel(testing_label_vector));
            out.nearest_neighbors(1:numel(nearest_neighbors),iNtrain,rt)=reshape(nearest_neighbors,1,numel(nearest_neighbors));
            
          end
          
        end
        out.gridParameters={'class';'number of training';'random spliting index'};
        out.gridParametersValuesCoarse{1}=1:max(testing_label_vector(:));
        out.gridParametersValuesCoarse{2}=listOfNtrain;
        out.gridParametersValuesCoarse{3}=1:nRandTrain;
        out.gridParametersValues{1}=1:max(testing_label_vector(:));
        out.gridParametersValues{2}=listOfNtrain;
        out.gridParametersValues{3}=1:nRandTrain;
        
        
        
        
      case 'kNN'
        
        in=classif_distance(feat_train,feat_train,classif_options);
        nRandTrain=getoptions(classif_options,'nRandTrain',5);
        Kmax=getoptions(classif_options,'Kmax',1);
        %listOfNtrain=[1,2,5,10,15,20];
        
        %out.listOfNtrain=listOfNtrain;
        
        for iNtrain=numel(listOfNtrain):-1:1
          Ntrain=listOfNtrain(iNtrain);
          for K=1:min(min(Kmax,numel(feat_train{1})),Ntrain);
            
            for rt=1:nRandTrain
              if (mod(rt,10)==1)
                fprintf('K %d Ntrain %d nrandomtrain %d \n',K,Ntrain,rt);
              end
              %compute the corresponding distance matrix and meta
              in2=classif_randDistanceMatFromDistanceMat(in,Ntrain);
              outclassif=classif_kNN(in2,K);
              %if (getoptions(classif_options,'store_everything',0))
              %    out.very_detailed_results{K}{iNtrain}{rt}=outclassif;
              %end
              %if (rt==1)
              %   meanconfusionmatrix{K}{iNtrain}=zeros(size(outclassif.confusionmatrix));
              %end
              %meanconfusionmatrix{K}{iNtrain}=meanconfusionmatrix{K}{iNtrain}+outclassif.confusionmatrix/nRandTrain;
              out.score_avg(iNtrain,rt,K)=outclassif.score_avg;
              out.score_per_cat(:,iNtrain,rt,K)=outclassif.score_per_cat;
              %out.training_indice_vector(1:numel(training_indice_vector),iNtrain,rt,ic,ig)=reshape(training_indice_vector,1,numel(training_indice_vector));
              %out.testing_indice_vector(1:numel(testing_indice_vector),iNtrain,rt,ic,ig)=reshape(testing_indice_vector,1,numel(testing_indice_vector));
              %out.predicted_label(1:numel(predicted_label),iNtrain,rt,ic,ig)=reshape(predicted_label,1,numel(predicted_label));
              %out.true_label(1:numel(testing_label_vector),iNtrain,rt,ic,ig)=reshape(testing_label_vector,1,numel(testing_label_vector));
              
              
            end
          end
        end
        
        out.gridParameters={'class';'number of training';'random spliting index';'K'};
        out.gridParametersValues{1}=1:numel(feat_train);
        out.gridParametersValues{2}=listOfNtrain;
        out.gridParametersValues{3}=1:nRandTrain;
        out.gridParametersValues{4}=1:Kmax;
        
        
      case 'nbnn'
        in=classif_distance_between_features(feat_train,feat_train,classif_options);
        nRandTrain=getoptions(classif_options,'nRandTrain',100);
        
        listOfNtrain=[5,10,20];
        out.listOfNtrain=listOfNtrain;
        for K=1:1
          for iNtrain=1:numel(listOfNtrain)
            Ntrain=listOfNtrain(iNtrain);
            for rt=1:nRandTrain
              if (mod(rt,10)==0)
                fprintf('K %d Ntrain %d nrandomtrain %d \n',1,Ntrain,rt);
              end
              %compute the corresponding distance matrix and meta
              in2=classif_randFeaturedistancefromFeaturedistance(in,Ntrain);
              outclassif=classif_nbnn(in2);
              if (getoptions(classif_options,'store_everything',0))
                out.very_detailed_results{K}{iNtrain}{rt}=outclassif;
              end
              if (rt==1)
                meanconfusionmatrix{K}{iNtrain}=zeros(size(outclassif.confusionmatrix));
              end
              meanconfusionmatrix{K}{iNtrain}=meanconfusionmatrix{K}{iNtrain}+outclassif.confusionmatrix/nRandTrain;
              out.score_avg(K,iNtrain,rt)=outclassif.score_avg;
              out.score_percat(K,iNtrain,rt,:)=outclassif.score_per_cat;
              
            end
          end
        end
        
      case 'PCA'
        nRandTrain=getoptions(classif_options,'nRandTrain',5);
        nImPerTrain = getoptions(classif_options,'nImPerTrain',1);
        %listOfNtrain=[1,2,5,10,15,20];
        %out.listOfNtrain=listOfNtrain;
        
        for iNtrain=numel(listOfNtrain):-1:1
          Ntrain=listOfNtrain(iNtrain);
          for rt=1:nRandTrain
            fprintf('\n PCA classif : Ntrain %d nrandomtrain %d ',Ntrain,rt);
            
            tilt= getoptions(classif_options,'tilt',0);
            sts = getoptions(classif_options,'sts',0);
            if tilt>=1
              % outclassif  = classif_pcamodel_alld_randomizedTraining_tilt( feat_train,Ntrain ,tilt);
              outclassif = classif_pcamodel_alld_randomizedTraining_tilt_packed( feat_train,Ntrain );
            else
              if sts==1
                outclassif = classif_subspaceToPca( feat_train,Ntrain );
              else
                outclassif = classif_pcamodel_alld_randomizedTraining( feat_train,Ntrain );
              end
            end
            %%out.score_avg{iNtrain}(:,rt)=outclassif.score_avg;
            dpca=numel(outclassif.score_avg);
            out.score_avg(iNtrain,rt,1:dpca)=outclassif.score_avg;
            %cell indx= number of training
            %first matrix index = dim of PCA
            %second matrxi index = num of random iteration
            %out.score_percat{iNtrain}(:,:,rt)=outclassif.score_per_cat;
            [dpca,nclass]=size(outclassif.score_per_cat);
            out.score_per_cat(:,iNtrain,rt,1:dpca)=reshape(outclassif.score_per_cat',[nclass,1,1,dpca]);
            
            %cell indx= number of training
            %first matrix index = dim of PCA
            %second matrxi index = class
            %third index = num of random iteration
            out.gridParametersValues{4}{iNtrain}=outclassif.dims;
          end
          %%out.PCAdims{iNtrain}=outclassif.dims;
          
          out.gridParameters={'class';'number of training';'random spliting index';'dim of PCA (as a function of number of training)'};
          out.gridParametersValues{1}=1:numel(feat_train);
          out.gridParametersValues{2}=listOfNtrain;
          out.gridParametersValues{3}=1:nRandTrain;
          %out.gridParemetersValues{4}=PCAdims;
          
        end
        
        
        
      case 'knnproj'
        nRandTrain=getoptions(classif_options,'nRandTrain',1);
        for iNtrain=numel(listOfNtrain):-1:1
          
          
          Ntrain=listOfNtrain(iNtrain);
          for rt=1:nRandTrain
            %randomly split data between training and testing
            [training_label_vector,training_instance_matrix,testing_label_vector,testing_instance_matrix]=classif_formatForSvmLibWithRandomTraining(feat_train,Ntrain);
            for K=1:Ntrain
              fprintf(' knnproject classif : Ntrain %d  rt %d K %d \n',Ntrain,rt,K);
              predicted_label= classif_knnprojection(K,training_label_vector,...
                training_instance_matrix,testing_instance_matrix);
              outclassif=classif_computeScore(predicted_label,testing_label_vector);
              
              out.score_avg(iNtrain,rt,K)=outclassif.score_avg;
              out.score_per_cat(:,iNtrain,rt,K)=outclassif.score_per_cat;
              out.confusionMatrix(:,:,iNtrain,rt,K)=outclassif.confusionMatrix;
              
            end
            
          end
        end
        out.gridParameters={'class';'number of training';'random spliting index';'K'};
        out.gridParametersValues{1}=1:numel(feat_train);
        out.gridParametersValues{2}=listOfNtrain;
        out.gridParametersValues{3}=1:nRandTrain;
        
      case 'SVM'
        nRandTrain=getoptions(classif_options,'nRandTrain',16);
        
        
        for iNtrain=numel(listOfNtrain):-1:1
          
          %first round. coars grid analysis for SVM.
          Ntrain=listOfNtrain(iNtrain);
          for rt=1:nRandTrain
            %randomly split data between training
            
            %launch libsvm with a range of parameters
            NsvmC=getoptions(classif_options,'NsvmC',16);
            NsvmG=getoptions(classif_options,'NsvmG',8);
            gridC=logspace(-8,8,NsvmC);
            gridGamma=logspace(-4,0,NsvmG);
            [training_label_vector,training_instance_matrix,testing_label_vector,testing_instance_matrix]=classif_formatForSvmLibWithRandomTraining(feat_train,Ntrain);
            for ic=1:numel(gridC)
              
              c=gridC(ic);
              for ig=1:numel(gridGamma)
                g=gridGamma(ig);
                tempo=sprintf('-g %g -c %g -h 0 -q', g, c);
                model = svmtrain(training_label_vector, training_instance_matrix,tempo );
                fprintf(' SVM classif : Ntrain %d  rt %d  ic %d ig %d ',Ntrain,rt,ic,ig);
                predicted_label = svmpredict(testing_label_vector, testing_instance_matrix, model );
                outclassif=classif_computeScore(predicted_label,testing_label_vector);
                
                out.score_avg_coarse(iNtrain,rt,ic,ig)=outclassif.score_avg;
                out.score_per_cat_coarse(:,iNtrain,rt,ic,ig)=outclassif.score_per_cat;
                out.confusionMatrix_coarse(:,:,iNtrain,rt,ic,ig)=outclassif.confusionMatrix;
                
              end
            end
          end
          
          
          %second round. finer grid analysis for SVM.
          %find the max
          inds=findmax(mean(out.score_avg_coarse(iNtrain,:,:,:),2));
          cMax=gridC(inds(3));
          gMax=gridGamma(inds(4));
          gridCFine=cMax*logspace(-2,2,NsvmC);
          gridGammaFine=gMax*logspace(-2,2,NsvmG);
          for rt=1:nRandTrain
            %randomly split data between training
            
            %launch libsvm with a range of parameters
            [training_label_vector,training_instance_matrix,...
              testing_label_vector,testing_instance_matrix,...
              training_indice_vector,testing_indice_vector]=...
              classif_formatForSvmLibWithRandomTraining(feat_train,Ntrain);
            for ic=1:numel(gridCFine)
              
              c=gridCFine(ic);
              for ig=1:numel(gridGammaFine)
                g=gridGammaFine(ig);
                tempo=sprintf('-g %g -c %g -h 0 -q', g, c);
                model = svmtrain(training_label_vector, training_instance_matrix,tempo );
                fprintf(' SVM classif FINE : Ntrain %d  rt %d  ic %d ig %d ',Ntrain,rt,ic,ig);
                predicted_label = svmpredict(testing_label_vector, testing_instance_matrix, model );
                outclassif=classif_computeScore(predicted_label,testing_label_vector);
                
                
                out.score_avg(iNtrain,rt,ic,ig)=outclassif.score_avg;
                out.score_per_cat(:,iNtrain,rt,ic,ig)=outclassif.score_per_cat;
                out.confusionMatrix(:,:,iNtrain,rt,ic,ig)=outclassif.confusionMatrix;
                out.training_indice_vector(1:numel(training_indice_vector),iNtrain,rt,ic,ig)=reshape(training_indice_vector,1,numel(training_indice_vector));
                out.testing_indice_vector(1:numel(testing_indice_vector),iNtrain,rt,ic,ig)=reshape(testing_indice_vector,1,numel(testing_indice_vector));
                out.predicted_label(1:numel(predicted_label),iNtrain,rt,ic,ig)=reshape(predicted_label,1,numel(predicted_label));
                out.true_label(1:numel(testing_label_vector),iNtrain,rt,ic,ig)=reshape(testing_label_vector,1,numel(testing_label_vector));
                %out.predicted_label(iNtrain,rt,ic,ig)
              end
            end
          end
          
        end
        out.gridParameters={'class';'number of training';'random spliting index';'SVM-C';'SVM-gamma'};
        out.gridParametersValuesCoarse{1}=1:max(testing_label_vector(:));
        out.gridParametersValuesCoarse{2}=listOfNtrain;
        out.gridParametersValuesCoarse{3}=1:nRandTrain;
        out.gridParametersValuesCoarse{4}=gridC;
        out.gridParametersValuesCoarse{5}=gridGamma;
        out.gridParametersValues{1}=1:max(testing_label_vector(:));
        out.gridParametersValues{2}=listOfNtrain;
        out.gridParametersValues{3}=1:nRandTrain;
        out.gridParametersValues{4}=gridCFine;
        out.gridParametersValues{5}=gridGammaFine;
        
        
        
        
      case 'voteSVM'
        nRandTrain=getoptions(classif_options,'nRandTrain',1);
        for iNtrain=numel(listOfNtrain):-1:1
          
          %first round. coars grid analysis for SVM.
          Ntrain=listOfNtrain(iNtrain);
          for rt=1:nRandTrain
            %randomly split data between training
            
            %launch libsvm with a range of parameters
            NsvmC=getoptions(classif_options,'NsvmC',16);
            NsvmG=getoptions(classif_options,'NsvmG',8);
            gridC=logspace(-8,8,NsvmC);
            gridGamma=logspace(-4,0,NsvmG);
            gridC=100000;
            gridGamma=10;
            
            
            gridC=1;
            gridGamma=0.1;
            
            %randomly split data into training and testing
            %but format it for frame handling
            nframe=size(feat_train{1}{1},1);
            ncat=numel(feat_train);
            
            fprintf('formating data for SVM classif \n');
            [training_label_vector,training_instance_matrix,testing_label_vector, ...
              testing_instance_matrix,training_indice_vector,testing_indice_vector]=...
              classif_formatForSvmLibWithRandomTrainingWindow(feat_train,10);
            
            %launch svm for a grid of parameters C and Gamma
            for ic=1:numel(gridC)
              
              c=gridC(ic);
              for ig=1:numel(gridGamma)
                g=gridGamma(ig);
                tempo=sprintf('-g %g -c %g -h 0 -q', g, c);
                fprintf('launching svm \n');
                model = svmtrain(training_label_vector, training_instance_matrix,tempo );
                fprintf(' SVM classif : Ntrain %d  rt %d  ic %d ig %d ',Ntrain,rt,ic,ig);
                predicted_label = svmpredict(testing_label_vector, testing_instance_matrix, model );
                
                %vote from predicted label
                predicted_label_vote=classif_vote_from_predicted_label(predicted_label,nframe,ncat);
                outclassif=classif_computeScore(predicted_label_vote,testing_label_vector(1:nframe:end));
                
                out.score_avg_coarse(iNtrain,rt,ic,ig)=outclassif.score_avg;
                out.score_per_cat_coarse(:,iNtrain,rt,ic,ig)=outclassif.score_per_cat;
                out.confusionMatrix_coarse(:,:,iNtrain,rt,ic,ig)=outclassif.confusionMatrix;
                out.predicted_label(1:numel(predicted_label_vote),iNtrain,rt,ic,ig)=reshape(predicted_label_vote,1,numel(predicted_label_vote));
              end
            end
            
          end
        end
        
        
      otherwise
        error('unknown classif algorithm');
    end
  otherwise
    error('randomized training value must be 0 or 1');
end
%out.bestscore=max(out.score);