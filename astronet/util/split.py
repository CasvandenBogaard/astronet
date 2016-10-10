
from nolearn.lasagne import TrainSplit
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold

def _sldict(arr, sl):
    
    if isinstance(arr, dict):
        return {k: v[sl] for k, v in arr.items()}
    else:
        return arr[sl]

class CustomTrainSplit(TrainSplit):
  
    def __init__(self, eval_size, cutoff, stratify=True):
    
        self.eval_size = eval_size
        self.stratify = stratify
        self.cutoff = cutoff
    
    def __call__(self, X, y, net):
        
        if not self.cutoff:
            if self.eval_size:
                
                if net.regression or not self.stratify:
                    kf = KFold(y.shape[0], round(1. / self.eval_size))
                else:
                    kf = StratifiedKFold(y, round(1. / self.eval_size))
      
                train_indices, valid_indices = next(iter(kf))
                X_train, y_train = _sldict(X, train_indices), y[train_indices]
                X_valid, y_valid = _sldict(X, valid_indices), y[valid_indices]
                
            else:
                
                X_train, y_train = X, y
                X_valid, y_valid = _sldict(X, slice(len(y), None)), y[len(y):]
  
            return X_train, X_valid, y_train, y_valid
    
        else:
            
            train_indices, valid_indices = range(self.cutoff), range(self.cutoff, len(y))
            X_train, y_train = _sldict(X, train_indices), y[train_indices]
            X_valid, y_valid = _sldict(X, valid_indices), y[valid_indices]
            
            return X_train, X_valid, y_train, y_valid
    

