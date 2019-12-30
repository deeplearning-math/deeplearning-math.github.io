from utils import *

class Params:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Model:
    def __init__(self, method, params=Params()):
        print("Initializing %s model..." %method)
        if method == 'LDA':
            self.method = 'LDA'
            self.model = init_LDA(params)
        elif method == 'LGR':
            self.method = 'LGR'
            self.model = init_LGR(params)
        elif method == 'SVM':
            self.method = 'SVM'
            self.model = init_SVM(params)
        elif method == 'RFC':
            self.method = 'RFC'
            self.model = init_RFC(params)
        else:
            print("Unsupported model!!!")
            self.method = ''
            self.model = None
    
    def train(self, X, y):
        assert (self.model is not None)
        print("Fitting the %s model..." %self.method)
        self.model.fit(X, y)
        print("Done!")

    def predict(self, X, y):
        assert (self.model is not None)
        return self.model.score(X, y), self.model.predict(X)
    
    def train_predict(self, X_train, y_train, X_test, y_test):
        self.train(X_train, y_train)
        return self.predict(X_test, y_test)