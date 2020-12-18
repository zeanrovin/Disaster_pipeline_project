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

# Load dataset from database 
db = sqlite3.connect('data/messages_categories.db')
cursor = db.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()[0][0]
df = pd.read_sql_query('SELECT * FROM '+tables,db)

print(df.head())

X = df['message']
y = df[df.columns[5:]]

print(y.head())

print(X.iloc[7], y.iloc[7])

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

print(X[7], '\n', tokenize(X[7]))

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    
    def starting_verb(self, text):
        
        # tokenize by sentence
        sentence_list = sent_tokenize(text)
        
        for sentence in sentence_list:
            
            # tokenize each sentence into words and tag part of speech
            pos_tags = pos_tag(tokenize(sentence))
            
            if len(pos_tags) != 0:
                # index pos_tags to get the first word and part of speech tag
                first_word, first_tag = pos_tags[0]

                # return true if the part of speech is an apporpriate verb
                if first_tag in ['VB','VBP']:
                    return True
            
        return False
        
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)
        
        return pd.DataFrame(X_tagged)

#print(StartingVerbExtractor(X[7]))

#Build Pipeline
def get_pipeline(clf=RandomForestClassifier()):
    
    pipeline = Pipeline([
                        ('features',FeatureUnion([
                                                 ('text-pipeline',Pipeline([
                                                                            ('vect', CountVectorizer(tokenizer= tokenize)),
                                                                            ('tfidf', TfidfTransformer())
                                                                           ])),
                                                 ('starting-verb',StartingVerbExtractor())
                                                 ])),
                        ('clf', MultiOutputClassifier(clf))
                        ])
    return pipeline

def get_fbeta_score(y_true, y_pred):

    """
    Compute F_beta score, the weighted harmonic mean of precision and recall

    Parameters
    ----------
    y : Pandas Dataframe
        y true
    y_pred : array 
        y predicted 

    Returns
    -------
    fbeta_score : float
    """
    score_list = []
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
        
    for index, col in enumerate(y_test.columns):
        error = fbeta_score(y_test[col], y_pred[:,index],1,average='weighted')
        score_list.append(error)
        
    fb_score_numpy = np.asarray(score_list)
    fb_score_numpy = fb_score_numpy[fb_score_numpy<1]
    fb_score = np.mean(fb_score_numpy)
    
    return fb_score

pipeline = get_pipeline()
print('pipeline get parameters \n',pipeline.get_params().keys())

X_train, X_test, y_train, y_test = train_test_split(X, y)

print('Shape of X training set {}'.format(X_train.shape), '|', 'Shape of y training set {}'.format(y_train.shape))
print('Shape of X testing set {}'.format(X_test.shape), '|', 'Shape of y testing set {}'.format(y_test.shape))


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred.shape,y_test.shape,len(list(y.columns))

overall_accuracy = (y_pred == y_test).mean().mean()
#fb_score = get_fbeta_score(y_test, y_pred)

print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
#print('Fbeta score {0:.2f}%\n'.format(fb_score*100))

#print(classification_report(y_test,y_pred))

y_pred_pd = pd.DataFrame(y_pred, columns = y_test.columns)
for column in y_test.columns:
    print('------------------------------------------------------\n')
    print('FEATURE: {}\n'.format(column))
    print(classification_report(y_test[column],y_pred_pd[column]))


#improving model with grid searcch

""" def build_model(clf = RandomForestClassifier()):
    
    pipeline = get_pipeline(clf)

        # specify parameters for grid search
    #parameters = {
    #                    'clf__estimator__min_samples_split': [2, 4],
    #                    'clf__estimator__criterion': ['log2', 'auto', 'sqrt', None],
    #                    'features__text-pipeline__tfidf__use_idf' : [True, False],
    #                    'clf__estimator__criterion': ['gini', 'entropy'],
    #                    'clf__estimator__max_depth': [None, 25, 50, 100, 150, 200],
    #    }

    parameters = {'vect__max_df': (0.5, 0.75, 1.0),
            'vect__ngram_range': ((1, 1), (1,2)),
            'vect__max_features': (None, 5000,10000),
            'tfidf__use_idf': (True, False)}
    #make_score= make_scorer(get_fbeta_score,greater_is_better=True)
        # create grid search object
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=-1)
    cv.fit(X_train,y_train)
    
    return cv

model = build_model() """


""" y_pred = model.predict(X_test)
overall_accuracy = (y_pred == y_test).mean().mean()

print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100)) """


pipeline = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(BernoulliNB()))])

parameters = {'vect__max_df': (0.5, 0.75, 1.0),
            'vect__ngram_range': ((1, 1), (1,2)),
            'vect__max_features': (None, 5000,10000),
            'tfidf__use_idf': (True, False)}

gs_clf = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)

y_pred = gs_clf.predict(X_test)

#joblib.dump(gs_clf.best_estimator_, 'bernoulli_best.pkl')
#print(classification_report(y_test, y_pred, target_names=y.columns))

y_pred_pd = pd.DataFrame(y_pred, columns = y_test.columns)
for column in y_test.columns:
    print('------------------------------------------------------\n')
    print('FEATURE: {}\n'.format(column))
    print(classification_report(y_test[column],y_pred_pd[column]))

joblib.dump(gs_clf.best_estimator_, 'bernoulli_best.pkl')
