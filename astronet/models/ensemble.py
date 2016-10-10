'''
Created on 11.08.2016

@author: fgieseke
'''

import copy
import numpy
from scipy.stats import mode

class VotingClassifier(object):
    """
    
    Parameters
    ----------
    estimators : list of (string, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        `self.estimators_`.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    """

    def __init__(self, estimators):

        self.estimators = estimators

    def fit(self, X, y):
        """ Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """


        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        self.estimators_ = []

        for clfname, clf in self.estimators:
            self.estimators_.append((clfname, copy.deepcopy(clf).fit(X, y)))

        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """


        allpreds = self._predict(X)
        
        preds, _ = mode(allpreds, axis=1)
        preds = preds[:,0]        

        return preds

    def predict_proba(self, X):
        
        return self._predict_proba(X)

    def get_params(self, deep=True):
        
        return {"estimators": self.estimators,
                }
    
    def set_params(self, **parameters):
        
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
            
        return self

    def _predict(self, X):
        """ Predict via individual estimators 
        """
        
        return numpy.asarray([clf.predict(X) for _, clf in self.estimators_]).T

    def _predict_proba(self, X):
        
        probs = numpy.asarray([clf.predict_proba(X) for _, clf in self.estimators_]) 
        return numpy.average(probs, axis=0, weights=None)    
    
if __name__ == '__main__':
    
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)])
    eclf1 = eclf1.fit(X, y)
    print(eclf1.predict(X))
    print(eclf1.predict_proba(X))
