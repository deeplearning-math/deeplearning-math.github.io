from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inear_model import LogisticRegression




def fit_data(inputs, labels, method):
    if method == 'LDA':
        classifier = LDA()
    if method == 'SVM':
        classifier = SVC()
    if method == 'random_forest':
        classifier = 
        
        
    classifier.fit(inputs, labels)
    
    return classifier

def predict_mnist(x, clf):
    clf.predict(x)
    
def predict_raphael(x, )



###### MNIST
model.fit(train_x, train_y, '...')
y = model.predict(test_x)

# compare y and test_y


###### Raphael
for i in range(num_img):
    test_x = image_features[i]
    test_y = labels[i]
    loocv_x, loocv_y
    
    model.fit(loocv_x, loocv_y)
    y = model.predict(test_x)
    
    # compare y and test_y
    
# average