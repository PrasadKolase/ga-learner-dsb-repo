# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
print(df.head())
print(df.info)

df['INCOME'].fillna(value='$0', inplace=True)
df['INCOME'] = df['INCOME'].apply(lambda x:str(x)[1:].replace(',',''))

df['HOME_VAL'].fillna(value='$0', inplace=True)
df['HOME_VAL'] = df['HOME_VAL'].apply(lambda x:str(x)[1:].replace(',',''))

df['BLUEBOOK'] = df['BLUEBOOK'].apply(lambda x:str(x)[1:].replace(',',''))
df['OLDCLAIM'] = df['OLDCLAIM'].apply(lambda x:str(x)[1:].replace(',',''))
df['CLM_AMT'] = df['CLM_AMT'].apply(lambda x:str(x)[1:].replace(',',''))

X = df.drop('CLAIM_FLAG', axis=1)
y = df['CLAIM_FLAG']

count = y.value_counts()
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)
# Code ends here


# --------------
# Code starts here
columns = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']

for i in columns:
    X_train[i] = X_train[i].astype(float)
    X_test[i] = X_test[i].astype(float)

for i in columns:
    print(X_train[i].isnull().sum())
    print(X_test[i].isnull().sum())
# Code ends here


# --------------
# Code starts here
X_train.dropna(axis=0, subset=['YOJ','OCCUPATION'], inplace=True)
X_test.dropna(axis=0, subset=['YOJ','OCCUPATION'], inplace=True)

y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

l = ['AGE','CAR_AGE','INCOME','HOME_VAL']
for i in l:
    X_train[i].fillna(value=X_train[i].mean(), inplace=True)
    X_test[i].fillna(value=X_test[i].mean(), inplace=True)
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for i in columns:
    le = LabelEncoder()
    X_train[i] = le.fit_transform(X_train[i].astype(str))
    X_test[i] = le.transform(X_test[i].astype(str))
# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model = LogisticRegression(random_state = 6)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state = 9)
X_train, y_train = smote.fit_sample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Code ends here


# --------------
# Code Starts here
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_test, y_pred)
print(score)
# Code ends here


