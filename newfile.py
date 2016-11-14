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
file_train=pd.read_csv("train1.csv")
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
#file=file[file.TARGET!='None']

print(file.shape)
label=file['TARGET']
feature=file.drop(['ID','TARGET'],axis=1)
col=feature.columns

##forest=RandomForestClassifier(n_estimators=100)
lr1=LogisticRegression(penalty='l1',C=0.1)
lr=LogisticRegression(penalty='l2')
lr2=LogisticRegression(penalty='l1')
sbmc=sk.svm.SVC(kernel='rbf')
##y_pred1=lr.fit(X_train,y_train).predict(X_test)
##y_pred2=forest.fit(X_train,y_train).predict(X_test)
##acc=(y_pred1==y_test).sum()/len(y_test)
##accfor=(y_pred2==y_test).sum()/len(y_test)
##print(acc)
##print(accfor)
##ar=feature.corr()
###print(ar.head())
##names=np.where(ar==1)
##print(names)
##print(names[0])
##ls=[]
##for i, j in zip(names[0],names[1]):
##    if i==j:
##        continue
##    else:
##        ls.append((i,j))
##
##print(len(ls))
##print(ls[:10])
##for i in ls[:int(len(ls)/2)]:
##    feature=feature.drop(feature.columns[i[0]],axis=1)
feature = preprocessing.normalize(feature, norm='l1')

trainx=pd.DataFrame(columns=col)
trainy=pd.DataFrame(columns=['Tar'])
testx=pd.DataFrame(columns=col)
testy=pd.DataFrame(columns=['Tar'])
X_train, X_test, y_train, y_test = train_test_split(feature,label, test_size=0.5)
forest=RandomForestClassifier(n_estimators=100)
forest1=AdaBoostClassifier(n_estimators=100)

##print(X_train[:10])
##print(y_train[:100])
etr=[]
etrlen=[]
ete=[]
etelen=[]
var=50
while var <len(X_train):
    inst=np.random.choice(range(len(X_train)),var)
    #print(inst)
    trainx=(X_train[inst])
    model2=lr2.fit(trainx,y_train.iloc[inst])
    pred=model2.predict(X_train)
    #print(y_train.iloc[inst])
    error=(pred!=y_train).sum()/len(y_train)
    #print(pred.sum())
    #error=recall_score(y_train,pred,average='binary')
    #print(classification_report(y_train,pred))
    etr.append(error)
    #print(etr)
    etrlen.append(var)
    var=var+500
var1=50
print("complete")
while var1 <len(X_test):
    inst1=np.random.choice(range(len(X_test)),var1)
    testx=(X_test[inst1])
    model2=lr2.fit(testx,y_test.iloc[inst1])
    pred1=model2.predict(X_test)
    error1=(pred1!=y_test).sum()/len(y_test)
    #error1=recall_score(y_test,pred1,average='binary')
    #print(classification_report(y_test,pred1))
    ete.append(error1)
    #print(ete)
    etelen.append(var1)
    var1=var1+500

    
    
    

##p=0
##while p<len(X_train):
##    for i,j in zip(X_train,y_train):
####    i=pd.DataFrame(i,columns=[col])
####    print(i)
##        trainx.loc[p]=i
##        #print(trainx)
##        
##        trainy.loc[p]=j
##        #print(trainy)
##        pred1=model2.predict(trainx)
##        #print(pred1)
##        errtrain=(trainy['Tar']!=pred1).sum()/len(trainy)
##        etr.append(errtrain)
##        #print(etr)
##        etrlen.append(len(trainx))
##        #print(etrlen)
##        p=p+1
##print("process complete")
##
##s=0
##while s<len(X_test):
##    for i,j in zip(X_test,y_test):
##        testx.loc[s]=i
##        testy.loc[s]=j
##        pred2=model2.predict(testx)
##        errtest=(testy['Tar']!= pred2).sum()/len(testy)
##        ete.append(errtest)
##        etelen.append(len(testx))
##        s=s+1
##print("process2 complete")



plt.plot(etrlen,etr,'r',lw=1,label='train')
plt.plot(etelen,ete,'b',lw=1,label='test')
plt.show()


##model1=lr1.fit(X_train,y_train)
##y_pred1=model1.predict(X_test)
##acc1=(y_pred1==y_test).sum()/len(y_test)
##print(acc1)
##model=lr.fit(X_train,y_train)
##y_pred=model.predict(X_test)
##acc=(y_pred==y_test).sum()/len(y_test)
##print(acc)
