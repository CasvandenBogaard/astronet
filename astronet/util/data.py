'''
Created on 31.05.2016

@author: fgieseke
'''

import numpy
from sklearn.cross_validation import KFold

def shuffle_all(L, n, seed=0):

    numpy.random.seed(seed)
    perm = numpy.random.permutation(n)
    for i in xrange(len(L)):
        L[i] = L[i][perm]

    return L
        
def _single_fold(n, train_size=0.5, test_size=0.5, shuffle=True, random_state=0):

    assert train_size + test_size <= 1.0
    
    allindices = numpy.array(range(n))
    
    if shuffle == True:
        numpy.random.seed(random_state)
        perm = numpy.random.permutation(n)
        allindices = allindices[perm]    
    
    train_range = allindices[:int(train_size * n)]
    test_range = allindices[int(train_size * n):int(train_size * n) + int(test_size * n)]

    yield (train_range, test_range)

def get_train_test_indices(n, train_size=0.5, test_size=0.5, shuffle=True, random_state=0):

    kf = _single_fold(n, train_size=train_size, test_size=test_size, shuffle=shuffle, random_state=random_state)
    
    for train_idx, test_idx in kf:
        return train_idx, test_idx
    
    
def get_folds(n, train_size=0.5, test_size=0.5, n_folds=1, shuffle=True, random_state=0):

    if n_folds > 1:
        kf = KFold(n, n_folds, shuffle=shuffle, random_state=random_state)
    else:
        kf = _single_fold(n, train_size=train_size, test_size=test_size, shuffle=shuffle, random_state=random_state)

    return kf
