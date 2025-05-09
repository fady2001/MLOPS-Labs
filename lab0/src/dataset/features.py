import numpy as np
import pandas as pd

from dataset.dataset import Dataset


def extract_features(df: pd.DataFrame) -> Dataset:
    # 1. Title Extraction from Name
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # 2. Family Size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # 3. Deck from Cabin
    df["Deck"] = df["Cabin"].astype(str).str[0]
    df["Deck"] = df["Deck"].fillna("U")

    # 4. Ticket Group Size
    ticket_counts = df["Ticket"].value_counts()
    df["TicketGroupSize"] = df["Ticket"].map(ticket_counts)

    # 5. Fare per Person
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df["FarePerPerson"].replace([np.inf, -np.inf], np.nan, inplace=True)

    # 6. Binning Age and Fare
    df["AgeBin"] = pd.cut(
        df["Age"], bins=[0, 12, 20, 40, 60, 80], labels=False, include_lowest=True
    )
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)

    # 7. Interaction Features
    df["Pclass*AgeBin"] = df["Pclass"] * df["AgeBin"].fillna(0).astype(int)
    df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)

    # 8. Missing Value Indicators
    df["CabinMissing"] = df["Cabin"].isnull().astype(int)
    df["AgeMissing"] = df["Age"].isnull().astype(int)

    return Dataset(df, target_col="Survived")
