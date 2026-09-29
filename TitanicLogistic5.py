#############################################################
# Date: 2026-08-16
# Topic:  Model Perservation (Logistic Regression)
# Key Learning: Logistic Regression - Load Data, Preprocessing (EDA), Split Dataset, Splitting,
#               Train and Test the model, Accuracy and Confusion Matrix Displayed
#               and PRESERVE model
#############################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib       # To Preserve the model

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#----------------------------------------------------------------
#   Function Name : LoadData
#   Description : Load Data from csv
#   Input : Name of csv file
#   Output : Data Frame
#   Author : Nidhi Girish Kadival
#   Date : 16/08/2026
#----------------------------------------------------------------

def LoadData(DataPath):

    df = pd.read_csv(DataPath)

    print("Dataset loaded successfully ")

    print(df.head())

    return df

#----------------------------------------------------------------
#   Function Name : PreprocessData
#   Description : Performs Data Analysis
#   Input : Dataframe
#   Output : Updated Dataframe
#   Author : Nidhi Girish Kadival
#   Date : 16/08/2026
#----------------------------------------------------------------

def PreprocessData(df):
    df = df.drop(
        columns = [
        "Passengerid",
        "zero"
        ],
        errors = "ignore")      # Ignores if a particular column name is not present in the dataset

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first=True,
        dtype=int
    )

    print(df.head(10))
    print("Data Preprocessing Completed")

    return df

#----------------------------------------------------------------
#   Function Name : SplitData
#   Description : It performs splitting activity
#   Input : df
#   Output : 4 subsets for training and testing
#   Author : Nidhi Girish Kadival
#   Date : 16/08/2026
#----------------------------------------------------------------

def SplitData(df):
    X = df.drop("Survived", axis = 1)
    Y = df["Survived"]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

    print("Data Splitting Sucessful")

    return X_train, X_test, Y_train, Y_test

def TrainModel(X_train,Y_train):
    model = LogisticRegression(max_iter=1000)

    model = model.fit(X_train,Y_train)

    print("Model Trained Succesfully")

    return model

def EvaluateModel(model, X_test, Y_test):
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test,Y_pred)

    print(f"Accuracy : {accuracy*100} %")

    cm = confusion_matrix(Y_test, Y_pred)

    print("Confusion Matrix : ")
    print(cm)

#----------------------------------------------------------------
#   Function Name : PreserveModel
#   Description : It performs model preservation into .pkl file
#   Input : model
#   Output : None
#   Author : Nidhi Girish Kadival
#   Date : 16/08/2026
#----------------------------------------------------------------

def PreserveModel(model, filename):
    joblib.dump(model, filename)

    print("Model Preseved with name ",filename)

#----------------------------------------------------------------
#   Function Name : main
#   Description : Entry Point function
#   Input : None
#   Output : None
#   Author : Nidhi Girish Kadival
#   Date : 16/08/2026
#----------------------------------------------------------------

def main():
    # Step 1 : Load Data
    df = LoadData("MarvellousTitanicDataset.csv")

    # Step 2 : Preprocessing
    df = PreprocessData(df)

    X_train, X_test, Y_train, Y_test = SplitData(df)

    model = TrainModel(X_train,Y_train)

    EvaluateModel(model, X_test, Y_test)

    PreserveModel(model,"MarvellousTitanic.pkl")

    

if(__name__ == "__main__"):
    main()