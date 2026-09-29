#############################################################
# Date: 2026-08-16
# Topic:  Advertising Case Study
# Key Learning: Load Data, Data Cleaning (EDA), Statistical Summary (describe()), Correlation Matrix
#               Splitting Data, Create and Train the Model, Testing, Evaluate the Model (MSE, RMSE, R2_Score)
#               Print Co-efficient of all and the Y Intercept
#############################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
def MarvellousRegression(DataPath):
    border = "-"*60
    
    # Step 1 : Load the data

    print(border)
    print("Step 1 : Load the data")
    print(border)

    df = pd.read_csv(DataPath)

    print(df.head())

    # Step 2 : Remove Unwanted Columns

    print(border)
    print("Step 2 : Remove Unwanted Columns")
    print(border)

    if("Unnamed: 0") in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(df.head())   

    # Step 3 : Check Missing Values

    print(border)
    print("Step 3 : Check Missing Values")
    print(border)    

    print("Total missing values : ")
    print(border)
    print(df.isnull().sum())
    print(border)     

    # Step 4 : Statistical Summary

    print(border)
    print("Step 4 : Statistical Summary")
    print(border) 

    print(df.describe())

    # Step 5 : Correalation

    print(border)
    print("Step 5 : Correalation")
    print(border)     

    print(df.corr())

    # Step 6 : Seperate Independent and Dependent Variables

    print(border)
    print("Step 6 : Seperate Independent and Dependent Variablesn")
    print(border)    

    X = df[["TV","radio","newspaper"]]
    Y = df["sales"]

    print("Independent Variables : ")
    print(X.head())
    print("Dependent Variables : ")
    print(Y.head())

    # Step 7 : Split Independent and Dependent Variables

    print(border)
    print("Step 7 : Split Independent and Dependent Variables")
    print(border)  

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

    print("Training Data : ",X_train.shape)
    print("Testing Data : ",X_test.shape)

    # Step 8 : Create and Train the Model

    print(border)
    print("Step 8 : Create and Train the Model")
    print(border)     

    model = LinearRegression()

    model = model.fit(X_train, Y_train)

    print("Model Trained Successfully!!")

    # Step 9 : Test the Model

    print(border)
    print("Step 9 : Test the Model")
    print(border)     

    Y_pred = model.predict(X_test)

    print("Expected Answers : ")
    print(Y_test[:3])

    print("Predicted Answers : ")
    print(Y_pred[:3])

    # Step 10 : Evaluate the Model

    print(border)
    print("Step 10 : Evaluate the Model")
    print(border)    

    MSE = mean_squared_error(Y_test,Y_pred)

    RMSE = np.sqrt(MSE)

    R2 = r2_score(Y_test, Y_pred)

    print("Mean Square Error : ",MSE)
    print("Root Mean Square Error : ",RMSE)
    print("R2 Score : ",R2)

    # Step 11 : Display Coefficients 

    print(border)
    print("Step 11 : Display Coefficients ")
    print(border)           

    print("Coefficient of TV : ",model.coef_[0])
    print("Coefficient of radio : ",model.coef_[1])
    print("Coefficient of newspaper : ",model.coef_[2])

    print("Intercept : ",model.intercept_)

def main():
    MarvellousRegression("Advertising.csv")

if(__name__ == "__main__"):
    main()    
