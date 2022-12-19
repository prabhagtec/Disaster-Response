import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
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
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    cat_df = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    cat_num=cat_df.sum()
    cat_names = list(cat_num.index)
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                {
                    "uid": "babced",
                    "fill": "tonexty",
                    "mode": "none",
                    "name": "Col2",
                    "type": "scatter",
                    "x": cat_names,
                    "y": cat_num,
                    "fillcolor": "rgb(224, 102, 102)"
                }
            ],
            'layout': {
                'title': 'Message scattered by Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                {
                    "uid": "f4de1f",
                    "hole": 0.4,
                    "name": "Genre",
                    "pull": 0,
                    "type": "pie",
                    "domain": {
                        "x": genre_names,
                        "y": genre_counts,
                    },
                    "marker": {
                        "colors": [
                            "#ffff99",
                            "#beaed4",
                            "#f0027f"
                        ]
                    },
                    "textinfo": "label+value",
                    "hoverinfo": "all",
                    "labels": genre_names,
                    "values": genre_counts
                }
            ],
            'layout': {
                'title': 'Messages by Genre'
            }
        }
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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()