'''
Created on 27.07.2016

@author: fgieseke
'''

import os
import sys
import json
import cPickle as pickle

def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)
            
def ensure_dir(f):
    
    d = os.path.dirname(f)
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except:
            pass

def store_results(data, ofname):
    
    ensure_dir(ofname)
    with open(ofname, 'w') as fp:
        json.dump(data, fp)    
    
def save_execution_file(odir):
    f = open(os.path.basename(sys.argv[0]),'r')

    ofname = os.path.join(odir, sys.argv[0])
    ensure_dir(ofname)
    
    with open(ofname, 'w') as ofile:
        ofile.write(f.read())        
        
def save_data(obj, fname):
    
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)
