#!/usr/bin/env python
# coding: utf-8

# Создадим оболочку SVM которая дополнительно обрабатывает данные, что бы использовать её в нескольких задачах

# In[3]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[4]:


class My_SVM():
    """
    SVM with LabelEncoder and StandardScaler
    """
    def __init__(self, *args, **kwargs):
        self.classifier = SVC(*args, **kwargs)
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
    def fit(self, X, Y):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''
        
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        self.scaler.fit(X)
        self.encoder.fit(Y.unique())
        return self.classifier.fit(self.scaler.transform(X), self.encoder.transform(Y))
    
    def predict(self, X):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        return: np.array (n, samples)
        Prediction of model
        '''
        pred_classes = self.classifier.predict(self.scaler.transform(X))
        return self.encoder.inverse_transform(pred_classes)
    
    def score(self, X, Y):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to score model
        Y : array like, shape = (n_samples, )
        Targets to score mode
        return score of model
        '''
        return self.classifier.score(self.scaler(X), self.encoder.transform(Y))
    
    def get_params(self, deep=True):
        """
        return params of model
        """
        return self.classifier.get_params(deep=deep)
    
    def set_params(self, **params):
        return self.classifier.set_params(**params)

