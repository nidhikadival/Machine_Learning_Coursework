#############################################################
# Date: 2026-08-23
# Topic: UniModel (Logistic Regression)
# Key Learning: Breast Cancer - Loading Dataset, Divide dataset into X,Y
#                               Splitting dataset, Scaling features, 
#                               Model Building & Evaluation
#############################################################

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

# Step 1 : load dataset

border = "-"*50

df = pd.read_csv("breast_cancer.csv")

print(border)
print("Shape of Dataset : ",df.shape)
print(border)

print(border)
print("First 5 records : ")
print(df.head())
print(border)

# Step 2 : Separate Features and Labels

X = df.drop("target",axis = 1)
Y = df["target"]

print(border)
print("X Shape : ",X.shape)
print("Y Shape : ",Y.shape)
print(border)

# Step 3 : Split Dataset into training and testing

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# Step 4 : Scaling the features

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

# Step 5 : Model Creation

model = LogisticRegression(max_iter=1000)

# Step 6 : Train the model

model = model.fit(X_train,Y_train)

# Step 7 : Test the model

Y_pred = model.predict(X_test)

# Step 8 : Evaluate the model

accuracy = accuracy_score(Y_test, Y_pred)

print(border)
print(f"Accuracy : {accuracy*100} %")
print(border)

print("Confusion Matrix : ")
print(confusion_matrix(Y_test,Y_pred))
print(border)