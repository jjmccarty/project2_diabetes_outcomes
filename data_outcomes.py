"""
Utilities to extractdata from the specified diabetes source csv and provides a 
cleaned dataframe. 
"""

# Dependencies
import pandas as pd

#Source file path for diabetes data
behavior_source_file_path = 'Resources/diabetes_data.csv'
data_source_file_path = 'Resources/diabetes.csv'
#source_file_path = 'Resources/Dataset_of_Diabetes.csv'

def getCleanColumnsForData(df):
    """
    Cleans the dataframe values.  Note this work inplace on the passed dataframe

    Args:
        df (dataframe): The dataframe to work on

    Returns:None
    """
    print(f'----> Renaming DiabetesPedigreeFunction column to FamilyHistory')
    df.rename(columns={'DiabetesPedigreeFunction':'FamilyHistory'}, inplace=True)

    print(f'----> Mean will be substituted for 0 values')
    subMeanForZero(df, 'Glucose')
    subMeanForZero(df, 'BloodPressure')
    subMeanForZero(df, 'SkinThickness')
    subMeanForZero(df, 'Insulin')
    subMeanForZero(df, 'BMI')

def subMeanForZero(df, column):
    """
    Substitutes the mean of of a column value for 0 values.  Note this currently
    works in place on the dataframe provided. 

    Args:
        df (dataframe): The dataframe to work on
        column (string): The column name to switch 0 values on 

    Returns: None
    """
    mean = df.loc[df[column] != 0][column].mean()       
    df[column] = df[column].apply( lambda x: int(mean) if x == 0 else int(x))
    print(f'----> {int(mean)} substituted for 0 values in {column}')

# Primary function to retrieve dataframe outsize of this python file
def getDiabetesDataDataframe():
    """
    Creates a savings account, calculates interest earned, and updates the account balance.

    Args:

    Returns:
        dataframe: The diabetes information as a cleaned dataframe
    """
    print(f'----> Retrieving information for {data_source_file_path}')
    df = pd.read_csv(data_source_file_path)
    getCleanColumnsForData(df)
    return df

def getCleanColumnsForBehavior(df):
    """
    Cleans the dataframe values.  Note this work inplace on the passed dataframe

    Args:
        df (dataframe): The dataframe to work on

    Returns:None
    """
    df.drop( df[ df['Sex'] == 1 ].index, inplace=True)

# Primary function to retrieve dataframe outsize of this python file
def getDiabetesBehaviorDataframe():
    """
    Creates a savings account, calculates interest earned, and updates the account balance.

    Args:

    Returns:
        dataframe: The diabetes information as a cleaned dataframe
    """
    print(f'----> Retrieving information for {behavior_source_file_path}')
    df = pd.read_csv(behavior_source_file_path)
    getCleanColumnsForBehavior(df)
    return df

if __name__ == '__main__':
    print('Run data_outcomes.py')

    df2 = getDiabetesBehaviorDataframe()
    print(df2)
