"""
Project: Disaster Response Pipeline (Udacity - Data Science NanoDegree)

Preprocessing of data steps

Sample Script Syntax
> run process_data.py <messages csv path> <categories csv path> <SQLite database path>

for example:
> run process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

"""


#import relevant libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load Data Function

    Parameters
    ----------
    messages_filepath : string -> Path to csv file for messages.
    categories_filepath : string -> Path to csv file for categories.

    Returns
    -------
    df : dataframe -> merged dataframe of messages and categories.

    """
    
    #load messages and categories
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge into one dataframe
    df = messages.merge(categories, how='outer', on=['id'])
    return df

    
def clean_data(df):
    """
    Clean Data Function
    
    Parameters
    ----------
    df : dataframe -> uncleaned dataframe containing merged messages and categories.

    Returns
    -------
    df : dataframe -> cleaned dataframe
    
    Cleaning steps:
        - columns split into separate categories and original aggregated column removed
        - column headings set to meaningful names
        - values set to one or zero for each category
        - duplicate rows removed

    """
    # create separate columns of data
    categories = df['categories'].str.split(';', expand = True)
    
    #get column names and apply them
    category_colnames = list(map(lambda x: x[ : -2], categories.iloc[0]))
    categories.columns = category_colnames
    
    # convert data to 1 or 0 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        # in some rows the value is 2 not 1, set to 1
        categories[column] = categories[column].map(lambda x: 1 if x > 1 else x)
        
    # drop the original categories column
    df.drop(['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicated rows
    df.drop_duplicates(inplace = True)
    
    return df
    

def save_data(df, database_filename):
    """
    Save Data Function

    Parameters
    ----------
    df : dataframe -> cleaned message and category dataframe
    database_filename : string -> name of SQLite database target

    Returns
    -------
    None.
    
    """
    #set engine and save dataframe to table
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('labelled_messages', engine, index=False, if_exists='replace')  


def main():
    """
    Main Function to orchestrate the ETL process
        1 - load message and category data from csv files 
        2 - clean data
        3 - save data to SQLite database

    Returns
    -------
    None.

    """
    # check for correct sys arguments and then execute pipeline
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()