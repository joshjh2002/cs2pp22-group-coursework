
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def changeColumnNames(df):
    """
    Renames specific columns in a DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The DataFrame with renamed columns.
    """
    df = df.rename(columns={"ApplicantIncome": "Applicant_Income", "CoapplicantIncome": "Coapplicant_Income",
                   "LoanAmount": "Loan_Amount", "Loan_Amount_Term": "Loan_Term"})
    return df

# Encoding categorical columns values to numerical values


def encodeColumns(df):
    """
    Encodes categorical columns in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the categorical columns to be encoded.

    Returns:
    pandas.DataFrame: The DataFrame with the categorical columns encoded.
    """

    # Encoding Gender Column
    df['Gender'].replace(['Male', 'Female'], [1, 0], inplace=True)
    # Encoding Married Column
    df['Married'].replace(['Yes', 'No'], [1, 0], inplace=True)
    # Encoding Self Employed Column
    df['Self_Employed'].replace(['Yes', 'No'], [1, 0], inplace=True)
    # Encoding Loan Status Column
    df['Loan_Status'].replace(['Y', 'N'], [1, 0], inplace=True)
    # Encoding Education Column
    df['Education'].replace(['Graduate', 'Not Graduate'], [1, 0], inplace=True)
    # Encoding Property Area
    df['Property_Area'].replace(['Semiurban', 'Urban', 'Rural'], [
                                0, 1, 2], inplace=True)

    return df

# 'Dependents' column has a value 3+ which is why the column is of object type.
# Need to change this value as we need to change the value so it can be of int type


def dependentsColumn(df):
    """
    Replaces the 'Dependents' column values in the given DataFrame with numeric values.

    Args:
        df (pandas.DataFrame): The DataFrame containing the 'Dependents' column.

    Returns:
        pandas.DataFrame: The DataFrame with the 'Dependents' column values replaced.
    """
    df['Dependents'].replace('3+', 3, inplace=True)
    return df


def dependentsColumn(df):

    df['Dependents'].replace('3+', 3, inplace=True)
    return df


def dropLoanID(df):
    """
    Drops the 'Loan_ID' column from the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame from which to drop the 'Loan_ID' column.

    Returns:
    pandas.DataFrame: The DataFrame with the 'Loan_ID' column dropped.
    """
    df = df.drop("Loan_ID", axis=1)
    return df


def dropLoanID(df):

    df = df.drop("Loan_ID", axis=1)

    return df

# Replacing Missing Values using KNN Imputer


def fillNans(df):
    """
    Fills the missing values in a DataFrame using the KNNImputer algorithm.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing missing values.

    Returns:
    pandas.DataFrame: The DataFrame with missing values filled using the KNNImputer algorithm.
    """
    imputer = KNNImputer(n_neighbors=1)
    df_after = imputer.fit_transform(df)

    # redefining df using the filled data frame
    df = pd.DataFrame(df_after, columns=df.columns)

    return df
