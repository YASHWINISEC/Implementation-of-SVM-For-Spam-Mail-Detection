# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: YASHWINI M
RegisterNumber:  212223230249
*/

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

## RESULT OUTPUT:
![282243022-28a5795a-2580-433a-9443-e2f07c687b5e](https://github.com/YASHWINISEC/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139361633/9936900b-0d54-497e-88d6-c5010e992011)

## data.head()
![326156121-aab424be-38f0-47ad-9df2-1244b4966cdc](https://github.com/YASHWINISEC/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139361633/1b78d483-4b56-42b8-95bd-97ba17f9213d)

## data.info()
![326156130-383da979-16ee-4c25-a35e-c9edb7b0d164](https://github.com/YASHWINISEC/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139361633/124e38eb-2e97-4ba5-ab76-701e94d57878)

## Y_prediction value:
![326156148-3d3d1c76-2a63-4e9c-a558-1b75216fdb57](https://github.com/YASHWINISEC/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139361633/b206bd4c-dd66-4df6-b946-49784ec0258c)

## Accuracy value:
![326156157-30f0f147-e396-4911-8922-d41da666e7ad](https://github.com/YASHWINISEC/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139361633/6f0f8423-e707-4135-9cdc-ef34d7f19a75)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
