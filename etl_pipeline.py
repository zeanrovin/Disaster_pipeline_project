# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from operator import itemgetter

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv('data/messages.csv')
print(messages.head())

# load categories dataset
categories = pd.read_csv('data/categories.csv')
print(categories.head())

# merge datasets
df = messages.merge(categories, on='id')
print(df.head())

# create a dataframe of the 36 individual category columns
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

print(categories.head())

for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1:]
    
    # convert column from string to numeric
    categories[column] = categories[column].astype('int')
print(categories.head())

print(df.head())

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis=1)
print(df.head())

# check number of duplicates
print(df[df.duplicated()])

# drop duplicates
print('shape before dropping duplicates', df.shape)
df = df.drop_duplicates()

print('shape after dropping duplicates', df.shape)

# check number of duplicates
print(df[df.duplicated()])


#drop table if exists
dbpath = 'sqlite:///data/messages_categories.db'
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

categories_1 = list(df.columns[5:])
categories_counts = list(df.iloc[:,5:].sum())

top_count = sorted(zip(categories_counts, categories_1), reverse=True)
#top_count = categories.nlargest(5,)

first_tuple_elements = []

for a_tuple in top_count:
	first_tuple_elements.append(a_tuple[0])

print(first_tuple_elements)

print(type(top_count))