import json
import plotly
import matplotlib.pyplot as plt
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.graph_objs as gobj
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/messages_categories.db')
df = pd.read_sql_table('messages_categories', engine)

# load model
model = joblib.load("models/model1.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    categories = list(df.columns[5:])
    categories_counts = list(df.iloc[:,5:].sum())

    top_count = sorted(zip(categories_counts, categories), reverse=True)[:5]
    bottom_count = sorted(zip(categories_counts, categories), reverse=False)[:5]
    #top_count = categories.nlargest(5,)

    first_tuple_elements = []
    second_tuple_elements = []

    for a_tuple in top_count:
        first_tuple_elements.append(a_tuple[0])
        second_tuple_elements.append(a_tuple[1])
    
    first_tuple_bottom = []
    second_tuple_bottom = []

    for b_tuple in bottom_count:
        first_tuple_bottom.append(b_tuple[0])
        second_tuple_bottom.append(b_tuple[1])

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }

    ]

    fig2 = gobj.Figure(gobj.Bar(
            x=categories_counts,
            y=categories,
            orientation='h',
            marker=dict(
            color='rgba(0,255,255, 0.2)',
            line=dict(color='rgba(0,255,255, 8.0)', width=3)
            )))

    fig3 = gobj.Figure(gobj.Bar(
            x=second_tuple_elements,
            y=first_tuple_elements,
            orientation='v',
            marker=dict(
            color='rgba(0,255,255, 0.2)',
            line=dict(color='rgba(0,255,255, 8.0)', width=3)
            )))

    fig4 = gobj.Figure(gobj.Bar(
            x=second_tuple_bottom,
            y=first_tuple_bottom,
            orientation='v',
            marker=dict(
            color='rgba(255,0,0, 0.2)',
            line=dict(color='rgba(255,0,0, 8.0)', width=3)
            )))

            

    graphs.append(fig2)
    graphs.append(fig3)
    graphs.append(fig4)
    

    """graphs.append(        {
            'data': [
                gobj.Bar(
                    
                    x=categories_counts,
                    y= categories,
                    orientation='h',
                    colors='crimson'
                )
            ],
            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Categories"
                },
                'xaxis': {
                    'title': "Count",
                }
            }
            
        }) """
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()