# %%
pwd

# %%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as plt

# %%
df=pd.read_csv('Data Sorting.csv', header = 0) 

# %%
df.head()

# %%
del df['Timestamp']

# %%
df.describe()


# %%
df.info()

# %%
df.shape

# %% [markdown]
# ## Missing value imputation

# %%
df['TWC4_Read'].mean()


# %%
df['TWC4_Read'].fillna(value = df['TWC4_Read'].mean(), inplace = True)

# %%
df['TWC1_Read'].fillna(value = df['TWC1_Read'].mean(), inplace = True)

# %%
df['TWC1_Expected'].fillna(value = df['TWC1_Expected'].mean(), inplace = True)

# %%
df['Variance'].fillna(value = df['Variance'].mean(), inplace = True)

# %%
df['TWC2_Read'].fillna(value = df['TWC2_Read'].mean(), inplace = True)

# %%
df['TWC2_Expected'].fillna(value = df['TWC2_Expected'].mean(), inplace = True)

# %%
df['Variance.1'].fillna(value = df['Variance.1'].mean(), inplace = True)

# %%
df['TWC3_Read'].fillna(value = df['TWC3_Read'].mean(), inplace = True)

# %%
df['TWC3_Expected'].fillna(value = df['TWC3_Expected'].mean(), inplace = True)

# %%
df['Variance.2'].fillna(value = df['Variance.2'].mean(), inplace = True)

# %%
df['TWC4_Expected'].fillna(value = df['TWC4_Expected'].mean(), inplace = True)

# %%
df['Variance.3'].fillna(value = df['Variance.3'].mean(), inplace = True)

# %%
df['TWCTR_Read'].fillna(value = df['TWCTR_Read'].mean(), inplace = True)

# %%
df['TWCTR_Expected'].fillna(value = df['TWCTR_Expected'].mean(), inplace = True)

# %%
df['Variance.4'].fillna(value = df['Variance.4'].mean(), inplace = True)

# %%
df['WTK_Read'].fillna(value = df['WTK_Read'].mean(), inplace = True)

# %%
df['WTK_Expected'].fillna(value = df['WTK_Expected'].mean(), inplace = True)

# %%
df['Variance.5'].fillna(value = df['Variance.5'].mean(), inplace = True)

# %%
df['WTU_Read(Truck)'].fillna(value = df['WTU_Read(Truck)'].mean(), inplace = True)

# %%
df['WTU_Expected(Truck)'].fillna(value = df['WTU_Expected(Truck)'].mean(), inplace = True)

# %%
df['Variance.6'].fillna(value = df['Variance.6'].mean(), inplace = True)

# %%
df.info()

# %% [markdown]
# # Dummy Variables

# %%
df.head()

# %%
df = pd.get_dummies(df, columns = ["Visual_Inspection", "Maintenance"], drop_first = True)

# %%
df.head()

# %% [markdown]
# ## X-Y Split

# %%
x = df.loc[:, df.columns != 'Maintenance_Yes']
type(x)

# %%
x.head()

# %%
x.shape

# %%
y = df['Maintenance_Yes']
type (y)

# %%
y.head()

# %%
y.shape

# %%
df.groupby(y).size()

# %% [markdown]
# ## Train test Split

# %%
from sklearn.model_selection import train_test_split

# %%
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

# %%
x_test.head()

# %%
x_train.head()

# %%
x_train.shape

# %%
x_test.shape

# %% [markdown]
# ## Standardizing Data

# %%
from sklearn.preprocessing import StandardScaler

# %%
df.head()

# %%
sc = StandardScaler().fit(x_train)

# %%
x_train_std = sc.transform(x_train)

# %%
x_test_std = sc.transform(x_test)

# %%
x_test_std

# %%
x_test

# %% [markdown]
# ## Training SVM classifier

# %%
from sklearn import svm

# %%
clf_svm_l = svm.SVC(kernel = 'linear', C = 0.01)
clf_svm_l.fit(x_train_std, y_train)

# %% [markdown]
# ## Predicting values using model

# %%
y_train_pred = clf_svm_l.predict(x_train_std)
y_test_pred = clf_svm_l.predict(x_test_std)

# %%
y_test_pred

# %% [markdown]
# ## Model Performance

# %%
from sklearn.metrics import accuracy_score, confusion_matrix

# %%
confusion_matrix(y_test,y_test_pred)

# %%
accuracy_score(y_test, y_test_pred)

# %%
clf_svm_l.n_support_

# %% [markdown]
# ### GridSearch
# Here I find best parameters

# %%
from sklearn.model_selection import GridSearchCV

# %%
params = {'C' : (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50)}

# %%
clf_svm_l = svm.SVC(kernel = 'linear')

# %%
svm_grid_lin = GridSearchCV(clf_svm_l, params, n_jobs = -1, 
                           cv = 10, verbose = 1, scoring = 'accuracy')

# %%
svm_grid_lin.fit(x_train_std, y_train)

# %%
svm_grid_lin.best_params_

# %%
linsvm_clf = svm_grid_lin.best_estimator_

# %%
accuracy_score(y_test, linsvm_clf.predict(x_test_std))

# %% [markdown]
# ### Polynomial Kernel
# #Finding performance of other algorithms for prediction

# %%
svm_clf_p3 = svm.SVC(kernel = 'poly', degree = 3, C = 0.1)

# %%
svm_clf_p3.fit(x_train_std, y_train)

# %%
y_train_pred = svm_clf_p3.predict(x_train_std)
y_test_pred = svm_clf_p3.predict(x_test_std)

# %%
accuracy_score(y_test, y_test_pred)

# %%
svm_clf_p3.n_support_

# %% [markdown]
# ### Radial kernel

# %%
clf_svm_r = svm.SVC(kernel = 'rbf', gamma = 0.5, C = 0.1)
clf_svm_r.fit(x_train_std, y_train)

# %%
y_train_pred = clf_svm_r.predict(x_train_std)
y_test_pred = clf_svm_r.predict(x_test_std)

# %%
accuracy_score(y_test, y_test_pred)

# %%
clf_svm_r.n_support_

# %% [markdown]
# ## Radial Grid

# %%
params = {'C' : (0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50),
         'gamma' : (0.001, 0.01, 0.1, 0.5, 1)}

# %%
clf_svm_r = svm.SVC(kernel = 'rbf')

# %%
svm_grid_rad = GridSearchCV(clf_svm_r, params, n_jobs = -1,
                           cv = 3, verbose = 1, scoring = 'accuracy')

# %%
svm_grid_rad.fit(x_train_std, y_train)

# %%
svm_grid_rad.best_params_

# %%
radsvm_clf = svm_grid_rad.best_estimator_

# %%
accuracy_score(y_test, radsvm_clf.predict(x_test_std))

# %%
import joblib
import time
import pickle

# %%
from sklearn import svm
import joblib

# %%
model = svm.SVC()
model.fit(x_train, y_train)

# %%
joblib.dump(model, 'model.pkl')

# %%



