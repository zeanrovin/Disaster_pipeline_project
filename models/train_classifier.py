import sys
import nltk
nltk.download(['punkt','wordnet','averaged_perceptron_tagger','stopwords'])

# import libraries
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import fbeta_score,classification_report,make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import sklearn.externals.joblib as extjoblib
import joblib
import re
import pandas as pd
import sqlite3
import numpy as np


def load_data(database_filepath):
    # Load dataset from database 
    db = sqlite3.connect('data/messages_categories.db')
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()[0][0]
    df = pd.read_sql_query('SELECT * FROM '+tables,db)

    X = df['message']
    y = df[df.columns[5:]]
    categor_names = df.columns[5:]

    return X, y, categor_names


def tokenize(text):
    # normalize text and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    words = [w for w in tokens if w not in stop_words]
    

    
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w) for w in words]
    
    return lemmed


def build_model(X, y):
    pipeline = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(BernoulliNB()))])
    
    parameters = {'vect__max_df': (0.5, 0.75, 1.0),
            'vect__ngram_range': ((1, 1), (1,2)),
            'vect__max_features': (None, 5000,10000),
            'tfidf__use_idf': (True, False)}


    gs_clf = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    gs_clf = gs_clf.fit(X_train, y_train)
    
    return gs_clf

def evaluate_model(model, X_test, y_test, category_names):
    #gs_clf = GridSearchCV(model, param_grid=parameters,n_jobs=-1)
    y_pred = model.predict(X_test)


    y_pred_pd = pd.DataFrame(y_pred, columns = y_test.columns)
    for column in y_test.columns:
        print('------------------------------------------------------\n')
        print('Category: {}\n'.format(column))
        print(classification_report(y_test[column],y_pred_pd[column]))

def save_model(model, model_filepath):
    joblib.dump(model.best_estimator_, model_filepath)


def main():
    
    database_filepath = 'data/messages_categories.db'
    model_filepath = 'models/model1.pkl'
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, y, category_names = load_data(database_filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print('Building model...')
    model = build_model(X, y)
    
    print('Training model...')
    model.fit(X_train, y_train)
    
    print('Evaluating model...')
    evaluate_model(model, X_test, y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    main()