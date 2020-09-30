# Analyze gas turbines parameters with machine learning

## calling libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
from sklearn import datasets
```
## importing database
```
gasturbine=pd.read_csv("file address.csv"))
```

## deleting model as a feature
```
gasturbine.rename(index=gasturbine.Model, inplace=True)
gasturbine.drop('Model', axis=1, inplace=True)
```



## heat map
```
num_var=gasturbine
corr=num_var.corr()
plt.figure(figsize = (16,16))
sb.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,vmin=-1,vmax=+1,cmap='coolwarm')
plt.show()
```


## Computing corrolation coefficient
```
from scipy.stats import spearmanr 
corr = gasturbine.corr(method='spearman')
```


## specifying source and target variables
```
sourcevars=gasturbine[['features']]
targetvar=gasturbine.feature
#x=sourcevars.iloc[0:268]
#y=targetvar.iloc[0:268]
x=sourcevars
y=targetvar
```
## preprocessing
```
from sklearn.preprocessing import minmax_scale
x=minmax_scale(x,feature_range=(0,1))
```



## dividing datas into test and train
```
from sklearn.model_selection import train_test_split
x_t, x_test, y_t, y_test=train_test_split(x, y, test_size=0.3, random_state=42) #, stratify=x)
```


## svr 
```
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
kernel_values=['rbf','sigmoid']
C_values=[10**(n-1) for n in range(1,5)]
esp_values=[0.02*n**2 for n in range(1,10)]
gammma_values=[0.01*n**2 for n in range(1,5)]
svm_params={'kernel':kernel_values,'C':C_values,'epsilon':esp_values,'gamma':gammma_values}
svr=SVR()
svr_cv=GridSearchCV(svr,svm_params,cv=5)
svr_cv.fit(x_t,y_t)
y_test_pred=svr_cv.predict(x_test)
y_train_pred=svr_cv.predict(x_t)
best_hyperparams=svr_cv.best_params_
best_accuracy=svr_cv.best_score_
from sklearn.metrics import r2_score
RSquared_test=r2_score(y_test,y_test_pred)
RSquared_train=r2_score(y_t,y_train_pred)
plt.figure(figsize=(5,5))
plt.scatter(y_t,y_train_pred,c='y',alpha=0.75,label='train data')
plt.scatter(y_test,y_test_pred,c='g',alpha=0.6,label='test data')
plt.title('SVR\nbest accuracy={}\nTest R-squared={}\nTrain R-squared={}\nbest hyperparams={}'.format(best_accuracy,RSquared_test,RSquared_train,best_hyperparams))
plt.xlabel('actual data')
plt.ylabel('predicted data')
plt.legend()
plt.plot([0,700000],[0,700000])
```


## knn reg
```
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(weights='uniform')
from sklearn.model_selection import GridSearchCV
params={'n_neighbors':range(1,30),'p':range(1,3)}
knn_model=GridSearchCV(knn,params,cv=5)
knn_model.fit(x_t,y_t)
best_hyperparams=knn_model.best_params_
best_accuracy=knn_model.best_score_
y_test_pred=knn_model.predict(x_test)
y_train_pred=knn_model.predict(x_t)
from sklearn.metrics import r2_score
RSquared_test=r2_score(y_test,y_test_pred)
RSquared_train=r2_score(y_t,y_train_pred)
plt.figure(figsize=(5,5))
plt.scatter(y_t,y_train_pred,c='y',alpha=0.75,label='train data')
plt.scatter(y_test,y_test_pred,c='g',alpha=0.6,label='test data')
plt.title('KNN\nbest accuracy={}\nTest R-squared={}\nTrain R-squared={}\nbest hyperparams={}'.format(best_accuracy,RSquared_test,RSquared_train,best_hyperparams))
plt.xlabel('actual data')
plt.ylabel('predicted data')
plt.legend()
plt.plot([0,700000],[0,700000])
```


## linear reg
```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_t,y_t)
y_test_pred=reg.predict(x_test)
y_train_pred=reg.predict(x_t)
accuracy=reg.score(x_test,y_test)
from sklearn.metrics import r2_score
RSquared_test=r2_score(y_test,y_test_pred)
RSquared_train=r2_score(y_t,y_train_pred)
plt.figure(figsize=(5,5))
plt.scatter(y_t,y_train_pred,c='y',alpha=0.75,label='train data')
plt.scatter(y_test,y_test_pred,c='g',alpha=0.6,label='test data')
plt.title('LINEAR\naccuracy={}\nTest R-squared={}\nTrain R-squared={}'.format(accuracy,RSquared_test,RSquared_train))
plt.xlabel('actual data')
plt.ylabel('predicted data')
plt.legend()
plt.plot([0,700000],[0,700000])
```

## poly reg
```
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
x_ = PolynomialFeatures(degree=2).fit_transform(x_t)
model=LinearRegression().fit(x_ , y_t)
x_test_ = PolynomialFeatures(degree=2).fit_transform(x_test)
y_test_pred=model.predict(x_test_)
y_train_pred=model.predict(x_ )
accuracy=model.score(x_test_,y_test)
from sklearn.metrics import r2_score
RSquared_test=r2_score(y_test,y_test_pred)
RSquared_train=r2_score(y_t,y_train_pred)
plt.figure(figsize=(5,5))
plt.scatter(y_t,y_train_pred,c='y',alpha=0.75,label='train data')
plt.scatter(y_test,y_test_pred,c='g',alpha=0.6,label='test data')
plt.title('POLY\naccuracy={}\nTest R-squared={}\nTrain R-squared={}'.format(accuracy,RSquared_test,RSquared_train))
plt.xlabel('actual data')
plt.ylabel('predicted data')
plt.legend()
plt.plot([0,700000],[0,700000])
```
