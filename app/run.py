import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
#import plotly.graph_objs as gro

import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

"""
Adding StartingVerbExtractor and tokeinze to run.py as per mentor suggested fix

"""
from sklearn.base import BaseEstimator, TransformerMixin

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor Class
    
    This class extracts the starting verb of a sentence, creating a new feature for the ML classifier
    """
    
    def starting_verb(self, text):
        """
        
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('labelled_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # category counts
    label_name = list(df.columns)[4:]
    label_count = df[label_name].sum()
    label_blank = df.shape[0] - label_count
    label_direct_count = df[df['genre']=='direct'][label_name].sum()
    label_news_count = df[df['genre']=='news'][label_name].sum()
    label_social_count = df[df['genre']=='social'][label_name].sum()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # Pie Chart - overview of what genre the messages have
        {
            'data': [{
                'values': genre_counts,
                'labels': genre_names,
                'type': 'pie',
                'hole': 0.5,
                }
            ],

            'layout': {
                'title': 'What type of messages are in the model training data?'
            }
        },
        
        # Bar Chart - distribution of labels in the training data
        {
            'data': [{
                'x': label_name,
                'y': label_count,
                'name': 'count of labels',
                'type': 'bar',
                'marker': {'line': {'color': 'black', 'width': 1}},
                'offsetgroup': 0,
                },
                {
                'name': 'blank',
                'x': label_name,
                'y': label_blank,
                'type': 'bar',
                'marker': {'color': 'white', 'line': {'color': 'black', 'width': 1}},
                'offsetgroup': 0,
                'base': label_count,
                }
            ],

            'layout': {
                'title': 'How are the labels distributed in the model training data?',
            }
        },
        
        # Stacked bar with  genre
        {
            'data': [{
                'x': label_name,
                'y': label_direct_count,
                'name': 'Direct',
                'type': 'bar',
                'marker': {'color': 'blue', 'line': {'color': 'black', 'width': 1}},
                'offsetgroup': 0,
                },
                {
                'name': 'News',
                'x': label_name,
                'y': label_news_count,
                'type': 'bar',
                'marker': {'color': 'green', 'line': {'color': 'black', 'width': 1}},
                'offsetgroup': 0,
                'base': label_direct_count,
                },
                {
                'name': 'Social',
                'x': label_name,
                'y': label_social_count,
                'type': 'bar',
                'marker': {'color': 'red', 'line': {'color': 'black', 'width': 1}},
                'offsetgroup': 0,
                'base': label_direct_count + label_news_count,
                }
            ],

            'layout': {
                'title': 'How are the labels distributed by Genre in the model training data?',
            }
        },       
    ]
    
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
    
    
    
    
