#############################################################
# Date: 2026-08-23
# Topic: UniModel (Decision Tree Regression)
# Key Learning: California Housing - ML Ops
#############################################################

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1 : Load data

border = "-"*50

df = pd.read_csv("california_housing.csv")

print(border)
print("Shape of dataset : ",df.shape)
print(border)

print(border)
print("First 5 records :")
print(df.head())
print(border)

# Step 2 : Separate Features and Labels

# X = df.drop(columns=["target"])
X = df.drop("target", axis=1)
Y = df["target"]

print(border)
print("X Shape : ",X.shape)
print("Y Shape : ",Y.shape)
print(border)

# Step 3 : Split dataset into training and testing

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# Step 4 : Create the model

model = DecisionTreeRegressor(random_state=42)

# Step 5 : Train the model

model = model.fit(X_train, Y_train)

# Step 6 : Test the model

Y_pred = model.predict(X_test)

# Step 7 : Evaluate the model

print(border)
print("MSE : ",mean_squared_error(Y_test, Y_pred))
print("R2 Score : ",r2_score(Y_test,Y_pred))
print(border)