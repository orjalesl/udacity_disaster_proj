import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load the data to the workspace to then be cleaned and pushed to the model.
    
    Inputs
    messages_filespath: str. the path to the messages.csv file
    categories_filepath: str. the path to the categories.csv file
    
    Output
    df - Dataframe combining both files provided"""
    
    # load the messages and categories files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge the datasets using the 'id' col
    df = messages.merge(categories,left_on='id',right_on='id')
    
    return df
    
    
    
def clean_data(df):
    """Clean the df provided to then be pushed to the model.
    
    Inputs
    df - original, unprocessed Dataframe
    
    Output
    df - cleaned/processed Dataframe """
    
    # split the categories out of the string
    df_categories = df['categories'].str.split(';',expand=True)
    
    # extract the first row to create column headers
    row = df_categories.iloc[0]

    # get everything in the string before the '-' and set the values
    # as the column names
    category_colnames = row.apply(lambda x: x.split('-')[0])
    df_categories.columns = category_colnames
    
    for column in df_categories:
        # set each value to be the last character of the string
        df_categories[column] = df_categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        df_categories[column] = df_categories[column].astype('int')
        
    # drop the original categories column from `df`
    df.drop(['categories'],axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,df_categories],axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # re,pve the values with '2.0' in 'related' column
    delete_rows = df[df['related'] == 2].index
    df.drop(index=delete_rows,axis=0,inplace=True)
    
    return df



def save_data(df, database_filename):
    """Save the dataframe to the specified database.
    
    Inputs
    df: cleaned Dataframe.
    database_filename: str. the path to the database
    
    Output
    df - Dataframe combining both files provided"""
    
    #load the df to the SQL table
    engine = create_engine('sqlite:///DisasterTextTable.db')
    df.to_sql(database_filename, engine, if_exists= 'replace', index=False)
    print(database_filename)


    
    
def main():
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