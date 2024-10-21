"""
Utilities to extractdata from the specified diabetes source csv and provides a 
cleaned dataframe. 
"""

# Dependencies
import pandas as pd

#Source file path for diabetes data
source_file_path = 'Resources/diabetes_data.csv'
#source_file_path = 'Resources/diabetes.csv'
#source_file_path = 'Resources/Dataset_of_Diabetes.csv'

def getCleanColumns(df):
    """
    Cleans the dataframe values.  Note this work inplace on the passed dataframe

    Args:
        df (dataframe): The dataframe to work on

    Returns:None
    """


# Primary function to retrieve dataframe outsize of this python file
def getDiabetesDataframe():
    """
    Creates a savings account, calculates interest earned, and updates the account balance.

    Args:

    Returns:
        dataframe: The diabetes information as a cleaned dataframe
    """
    print(f'----> Retrieving information for {source_file_path}')
    df = pd.read_csv(source_file_path)
    getCleanColumns(df)
    return df

if __name__ == '__main__':
    print('Run data_outcomes.py')

    df = getDiabetesDataframe()
    print(df)
