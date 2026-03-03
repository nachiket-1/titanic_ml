import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def engineer_features(df):
    df = df.copy()

    # Extract title from Name
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"], "Rare"
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Age & Fare bins
    df["AgeBand"] = pd.cut(df["Age"].fillna(df["Age"].median()), 5,
                           labels=["Child","Young","Adult","MiddleAge","Senior"])
    df["FareBand"] = pd.qcut(df["Fare"].fillna(df["Fare"].median()), 4,
                              labels=["Low","Mid","High","VeryHigh"])

    # Cabin known flag
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df


def get_feature_columns():
    numeric = ["Age","Fare","SibSp","Parch","FamilySize","IsAlone","HasCabin"]
    categorical = ["Pclass","Sex","Embarked","Title","AgeBand","FareBand"]
    return numeric, categorical


def build_preprocessor():
    numeric, categorical = get_feature_columns()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical)
    ])


def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    df = engineer_features(df)
    numeric, categorical = get_feature_columns()
    X = df[numeric + categorical]
    y = df["Survived"]
    return X, y