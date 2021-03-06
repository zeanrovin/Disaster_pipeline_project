import sys

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    print(messages.head())

    #load categories dataset
    categories = pd.read_csv(categories_filepath)
    print(categories.head())
    
    # merge datasets
    df = messages.merge(categories, on='id')

    return df

def clean_data(df):
    #Seperate the categories for cleaning
    categories = df['categories'].str.split(';', expand=True)
    print(categories.head())

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    print(row)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames =  row.apply(lambda x: x[:-2])
    print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames

    categories.columns = category_colnames

    print(categories.head())

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    print(categories.head())

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    print(df.columns)
    print("after concat:", df)

    #drop the categories column as it no longer makes any purpose
    df = df.drop(columns = ['categories'])

    #'related' column has 3 unique values[0,1,2], we'll treat 2 as 1
    df['related'].replace({2 : 1})

    # check number of duplicates
    print(sum(df.duplicated()))

    # drop duplicates
    print('shape before dropping duplicates', df.shape)
    df = df.drop_duplicates()

    print('shape after dropping duplicates', df.shape)

    # check number of duplicates
    print(sum(df.duplicated()))

    return df


def save_data(df, database_filename):
    #save the cleaned dataframe into a db file
    #drop table if exists
    dbpath = 'sqlite:///{}'.format(database_filename)
    table = 'messages_categories'
    engine = create_engine(dbpath)
    connection = engine.raw_connection()
    cursor = connection.cursor()
    command = "DROP TABLE IF EXISTS {};".format(table)
    cursor.execute(command)
    connection.commit()
    cursor.close()

    engine = create_engine(dbpath)
    df.to_sql(table, engine, index=False)  


def main():
    """"
    Input in the cmd: 'python data/process_data.py data/messages.csv data/categories.csv 
    data/messages_categories.db'
'
    ***[1] python data/process_data.py - to run the program
    ***[2] data/messages.csv: CSV file to be ETL
    ***[3] data/categories.csv: CSV file to be ETL
    ***[4] data/messages_categories.db: Output of the this ETL program(process_data.py)
    """
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