import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as auc
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import statsmodels.api as smf
forest=RandomForestClassifier(n_estimators=100)
file_train=pd.read_csv("train.csv")
colzerostd=[col  for col in file_train.columns if file_train[col].std()==0]
file=file_train.drop(colzerostd, axis=1)
coldup=[]
print(file.shape)
for i in range(len(file.columns)-1):
    for j in range(i+1,len(file.columns)):
        if np.array_equal(file[file.columns[i]].values,file[file.columns[j]].values):
            coldup.append(file.columns[j])
            
file=file.drop(coldup,axis=1)
print(file.shape)

label=file['TARGET']
feature=file.drop(['ID','TARGET'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(feature,label, test_size=0.30)

newcsv=pd.DataFrame(index=range(len(X_train.columns)),columns=['feat','auc'])
for k,feat in enumerate(X_train.columns):
    newtrainfeat=X_train[feat].values.reshape(-1,1)
    newvalidfeat=X_test[feat].values.reshape(-1,1)
    
    forest.fit(newtrainfeat,y_train)
    Auc=auc(y_test,forest.predict_proba(newvalidfeat)[:,1])
    newcsv.ix[k,'auc']=Auc
    newcsv.ix[k,'feat']=feat
newcsv=newcsv.sort_values(by='auc',axis=0,ascending=False).reset_index(drop=True)
print(newcsv.head(10))
###combination of features
featcombo=newcsv.ix[0:20,'feat']
print(featcombo)
featcomboname=[feat+str(x+1) for x in range(5)]
newcsvv=pd.DataFrame(index=range(300),columns=featcomboname+['newauc'])
ft=RandomForestClassifier(n_estimators=30)
for i in range(300):
    featselect=np.random.choice(len(featcombo),5,replace=False)
    featselectname=[featcombo[x] for x in featselect]
    for j in range(len(featselectname)):
        newcsvv.ix[i,featcomboname[j]]=featselectname[j]
    trainfeat=X_train.ix[:,featselectname]
    testfeat=X_test.ix[:,featselectname]
    ft.fit(trainfeat,y_train)
    testauc=auc(y_test,ft.predict_proba(testfeat)[:,1])
    newcsvv.ix[i,'newauc']=testauc
newcsvv=newcsvv.sort_values(by='newauc',axis=0,ascending=False).reset_index(drop=True)
print(newcsvv.head(10))                
#hh=pca.fit_transform(X_test)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_test,y_test, test_size=0.50)
##lkm=smf.Logit(y_train1,X_train1)
##mod=lkm.fit()
##print(mod.summary)
X_test1=scale(X_test1)
X_train1=scale(X_train1)
pca=PCA(n_components=0.98).fit(X_train1)
X_train1pca=pca.transform(X_train1)
X_test1pca=pca.transform(X_test1)

#hh1=pca.fit_transform(X_train1)
   
##print(file_train.head())
##file_attr=file_train.ix[:,:-1]

##file_label=file_train.ix[:,-1]
##X_train, X_test, y_train, y_test = train_test_split(file_attr,file_label, test_size=0.30)
lr=LogisticRegression(penalty='l2')
#print(np.linalg.matrix_rank(hh.values))

##y_pred1=mod.predict(hh1)
y_pred1=lr.fit(X_train1,y_train1).predict(X_test1)
y_pred2=lr.fit(X_train1pca,y_train1).predict(X_test1pca)

###sbmc=sk.svm.SVC()
#y_pred=lr.fit(X_train,y_train).predict(X_test)
#y_pred1=forest.fit(X_train,y_train).predict(X_test)
##print(y_pred.shape)
##print(y_test.shape)
accpca=(y_pred2==y_test1).sum()/len(y_test1)
acc=(y_pred1==y_test1).sum()/len(y_test1)
print(accpca,acc)
print(accpca+acc)

#print(mod.summary())
#print((y_pred1==y_test).sum()/len(y_test))

##print(roc_auc_score(y_test,y_pred))
