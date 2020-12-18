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
    print(df.head())

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

    messages_filepath = 'data/messages.csv'
    categories_filepath = 'data/categories.csv' 
    database_filepath = 'data/messages_categories.db'

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
            .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)
    
    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)
    
    print('Cleaned data saved to database!')

    """ print('Please provide the filepaths of the messages and categories '\
            'datasets as the first and second argument respectively, as '\
            'well as the filepath of the database to save the cleaned data '\
            'to as the third argument. \n\nExample: python process_data.py '\
            'disaster_messages.csv disaster_categories.csv '\
            'DisasterResponse.db') """


if __name__ == '__main__':
    main()