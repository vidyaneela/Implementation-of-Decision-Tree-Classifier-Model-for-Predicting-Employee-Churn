# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2.Read the data set.
3.Apply label encoder to the non-numerical column inoreder to convert into numerical values.
4.Determine training and test data set.
5.Apply decision tree Classifier and get the values of accuracy and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: M.Vidya Neela
RegisterNumber:  212221230120
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![decision tree classifier model](sam.png)
![200617343-4b606eaf-6918-46b1-a7ef-91a49bf4e4fc](https://user-images.githubusercontent.com/94169318/204431272-09725dbe-97b0-46f8-9593-573a0ef9b9c5.png)
![200617442-2d8a2866-06b5-4717-952b-c75a55b73886](https://user-images.githubusercontent.com/94169318/204431311-c79110ea-50ec-49d7-92ca-49920f4083b1.png)
![200617478-f81bd031-ccdd-4d22-8295-758384740aa5](https://user-images.githubusercontent.com/94169318/204431344-c38de66d-60d2-4a67-8a62-1389a18b315b.png)
![200617634-e069520e-33b3-4015-8a77-4cfe9c47b53c](https://user-images.githubusercontent.com/94169318/204431426-4f73c285-2c96-4d04-9efc-f7edf5be0af8.png)
![200617666-daff2eab-0306-453c-94c6-4456beb9565d](https://user-images.githubusercontent.com/94169318/204431448-c31efe7f-4f6b-4110-9fb2-cdf7d7d2785f.png)
![200617788-622864f2-ae41-4327-b5c9-776ebcad87b7](https://user-images.githubusercontent.com/94169318/204431514-644b2673-f477-4e43-9bc6-9b98a8ba4ffe.png)
![200617741-207d4daa-627a-4ddd-99b0-4bdfd63d5365](https://user-images.githubusercontent.com/94169318/204431477-5c78000d-73c5-414a-9ee2-4e2c65d6a7a6.png)
![200617830-127c32a2-fb2a-48b7-87ad-391705a0f589](https://user-images.githubusercontent.com/94169318/204431538-c372cce1-5b37-400d-a8e4-fa729f541267.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
