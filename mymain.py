import time
import pandas as pd
import numpy as np
import xgboost as xgb

import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning) # suppress annoying numpy warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# apply column transformations to a dataframe  
def process(data):        
    data['emp_length'] = data.emp_length.fillna('Unknown').str.replace('<','LT')
    data['dti'] = data.dti.fillna(0)
    data['revol_util'] = data.revol_util.fillna(0)
    data['mort_acc'] = data.mort_acc.fillna(0)
    data['pub_rec_bankruptcies'] = data.pub_rec_bankruptcies.fillna(0)
    temp = pd.to_datetime(data.earliest_cr_line)
    data['earliest_cr_line'] = temp.dt.year*12 - 1950*12 + temp.dt.month
    data.drop(['emp_title','title','zip_code','grade','fico_range_high'], axis=1, inplace=True)
    return data

# create test and train dataframes in format for modelling
def prep_train_test(train, test):    
    train = process(train)                      # apply data fixes and transformations to train dataframe
    X_train = train.drop(['loan_status'], axis=1) 
    X_train = pd.get_dummies(X_train)           # create dataframe with dummy variables replacing categoricals
    X_train = X_train.reindex(sorted(X_train.columns), axis=1) # sort columns to be in same sequence as test
    y_train = (train.loan_status!='Fully Paid').astype(int) # convert loan status to binary 0/1

    test = process(test)                        # apply data fixes and transformations to test dataframe
    X_test = pd.get_dummies(test)               # create dataframe with dummy variables replacing categoricals

    all_columns = X_train.columns.union(X_test.columns)      # add columns to test that are in train but not test
    X_test = X_test.reindex(columns=all_columns).fillna(0)
    X_test = X_test.reindex(sorted(X_train.columns), axis=1) # sort columns to be in same sequence as train
    return X_train, y_train, X_test


##### START OF MAIN ####
print('Reading and processing data files...')
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
seed = 42 # ensure reproducibility

X_train, y_train, X_test = prep_train_test(train, test)  # pre-process test and train

models = [
    LogisticRegression(penalty='l1',C=1, random_state=seed),
    
    GradientBoostingClassifier(max_features='sqrt', learning_rate=0.055, n_estimators=780, max_depth=7, 
                               min_samples_leaf=2, subsample=0.9, min_samples_split=4,
                               min_weight_fraction_leaf=0, random_state=seed),
    
    xgb.XGBClassifier(learning_rate=0.037, n_estimators=860, min_child_weight=8, max_depth=7, gamma=0.3,
                       subsample=0.52, colsample_bytree=0.92, reg_lambda=0.67, reg_alpha=0.03,  
                       objective= 'binary:logistic', n_jobs=-1, random_state=seed, eval_metric='logloss'),
]

print('Starting Modelling...')
for i, model in enumerate(models):   # loop through three models, fitting, predicting and writing submission file
    start_time = time.time()
    _ = model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:,1]
    df = pd.DataFrame({'id': test.id, 'prob': probs.round(5)})
    df.to_csv('mysubmission'+str(i+1)+'.txt', index=False)
    print('Created mysubmission'+str(i+1)+'.txt, rows=', df.shape[0],'in', round(time.time()-start_time,2),'secs') 
