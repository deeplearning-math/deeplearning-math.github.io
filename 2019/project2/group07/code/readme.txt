0) dataset:crop the paintings into small patches with size of 224*224, the number of training set:5000, the number of test set:2104.
After training and validation, we predict the identity of 7 disputed images.
1) use feature extraction.py to extract the 4096 feature vector by pretrained VGG16 model
2) use transfer learning.py to fine-tune the last layer of VGG16 to do image classification directly
3) use KNN.py to find the best K parameter by cross-validation based on feature vector
4) use Four traditional supervisde learning methods.py to do classification and plot the ROC curve 


