#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


# In[11]:


def get_data_head(df):
    return df.head()


# In[2]:


def get_data_info(df):
    return df.info()


# In[3]:


def get_missing_values_df(df):
    """
    Calculates the number and percentage of missing values in each column of a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing the number and percentage of missing values for each column,
                          sorted by the number of missing values in descending order.
    """

    # Calculate the number of missing values per column
    missing_values_count = df.isnull().sum()

    # Calculate the percentage of missing values
    missing_values_percentage = (df.isnull().sum() / len(df)) * 100

    # Combine both counts and percentages into a DataFrame
    missing_values_df = pd.DataFrame({'Number of Missing Values': missing_values_count,
                                      'Percentage of Missing Values': missing_values_percentage})

    # Sort the DataFrame by the number of missing values, descending
    missing_values_df = missing_values_df[missing_values_df['Number of Missing Values'] > 0].sort_values(
        by='Number of Missing Values', ascending=False)

    return missing_values_df


# In[4]:


def get_data_description(df):
    return df.describe()


# In[5]:


def get_loan_status_dataframe(df):
    """
    Calculate the count and percentage of each Loan_Status in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the Loan_Status column.

    Returns:
    pandas.DataFrame: A DataFrame with two columns - 'Count' and 'Percentage',
                      representing the count and percentage of each Loan_Status.
    """
    # Calculate the count of each Loan_Status
    loan_status_count = df['Loan_Status'].value_counts()

    # Calculate the percentage of each Loan_Status
    loan_status_percentage = (df["Loan_Status"].value_counts() / len(df) * 100)

    # Combine both counts and percentages into a DataFrame
    loan_status_df = pd.DataFrame({'Count': loan_status_count,
                                   'Percentage': loan_status_percentage})

    return loan_status_df


# In[6]:


def create_loan_status_countplot(df):
    """
    Create a countplot to visualize the distribution of loan status.

    Parameters:
    df (DataFrame): The input DataFrame containing loan data.

    Returns:
    plt (matplotlib.pyplot): The matplotlib plot object.
    """

    # Initialise figure
    plt.figure()

    # Create the countplot
    sns.countplot(x='Loan_Status', data=df)

    # Set title and axis labels
    plt.title("Distribution of Loan Status")
    plt.xlabel("Loan_Status")
    plt.ylabel("Count")

    return plt


# In[7]:


def create_loan_amount_boxplot(df):
    """
    Create a boxplot to visualize the distribution of loan amounts based on loan status.

    Parameters:
    df (DataFrame): The input DataFrame containing loan data.

    Returns:
    plt (matplotlib.pyplot): The generated boxplot.

    """
    # Initialise figure
    plt.figure()

    # Create the boxplot
    sns.boxplot(x='Loan_Status', y='LoanAmount', data=df)

    return plt


# In[8]:


def create_loan_amount_violinplot(df):
    """
    Create a violin plot to visualize the distribution of loan amounts across different property areas.

    Parameters:
    df (DataFrame): The input DataFrame containing the loan data.

    Returns:
    plt (matplotlib.pyplot): The matplotlib.pyplot object containing the violin plot.
    """

    # Initialise figure
    plt.figure()

    # Creating violin plot for 'Property_Area' vs 'LoanAmount'
    sns.violinplot(x='Property_Area', y='LoanAmount', data=df)

    # Display the plots
    plt.tight_layout()

    return plt


# In[9]:


def label_encode_binary_categorical_columns(df):
    """
    Encodes binary categorical columns in a DataFrame using label encoding.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the binary categorical columns.

    Returns:
    pandas.DataFrame: The DataFrame with the binary categorical columns encoded using label encoding.
    """

    label_encoder_columns = ['Gender', 'Married', 'Education',
                             'Self_Employed', 'Property_Area', 'Loan_Status']

    # The scikit-learn's LabelEncoder encodes categorical columns into numerical format
    label_encoder = LabelEncoder()

    # Iterate over each categorical column and transform the data using label encoding.
    for col in label_encoder_columns:
        df[col] = label_encoder.fit_transform(df[col])

    return df


# In[10]:


def get_correlation_matrix(df):
    """
    Calculate and visualize the correlation matrix of a dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    seaborn.heatmap: A heatmap visualization of the correlation matrix.
    """

    # Set up the figure size for the heatmap visualization
    plt.figure(figsize=(10, 8))

    # Calculate the correlation matrix which gives us pairwise correlation of all columns in the dataframe.
    corr_df = df.corr()

    # Use seaborn's heatmap function to plot the correlation matrix with annotations.
    return sns.heatmap(corr_df, annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm')
