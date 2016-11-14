import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import statsmodels.api as smf
file_train=pd.read_csv("train.csv")
print(file_train.head())
print(file_train.shape)
colzerostd=[col  for col in file_train.columns if file_train[col].std()==0]
file=file_train.drop(colzerostd, axis=1)
coldup=[]
print(file.shape)
for i in range(len(file.columns)-1):
    for j in range(i+1,len(file.columns)):
        if np.array_equal(file[file.columns[i]].values,file[file.columns[j]].values):
            coldup.append(file.columns[j])
file=file.drop(coldup,axis=1)
file0=pd.DataFrame()
file1=pd.DataFrame()
file01=pd.DataFrame()
print(file.shape)
##for i in range(len(file)):
##    print(file.iloc[i,-1])
for i in range(len(file)):
    if int(file.iloc[i,-1])!=0:
        file1.append(file.iloc[i])
    else:
        file0.append(file.iloc[i])
print(file0.shape)
print(file1.shape)
size=file1.shape[0]
file01index=np.random.choice(range(len(file0)),size,replace=False)
file01=file0.iloc[file01index]
file0=file0.drop(file0.iloc[file01index])
newfile=concat([fil01,file1],ignore_index=True)
newfile=newfile.iloc[np.random.permutation(len(newfile))]
label=newfile['TARGET']
feature=newfile.drop(['ID','TARGET'],axis=1)
feature = preprocessing.normalize(feature, norm='l1')
X_train, X_test, y_train, y_test = train_test_split(feature,label, test_size=0.5)
