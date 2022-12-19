import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, precision_score
from sklearn.tree import DecisionTreeClassifier
import pickle

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine) 
    #Define feature and target variables X and Y
    X = df.message.values
    Y = df.iloc[:, 4:40]
    category_names = Y.columns
    return X,Y, category_names
    pass


def tokenize(text):
    #tokenization function to process text data
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer() 

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = WordNetLemmatizer().lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    pass


def build_model():
    #This Machine pipeline model uses RandomForestClassifier in MultiOutputClassifier to predict multiple target variables.
    pipeline =  Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # specify parameters for grid search
    parameters = {
        'clf__estimator__random_state': [35],
        'clf__estimator__n_estimators': [10,15]
    }
    # create grid search object
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    #predicting model
    Y_pred = model.predict(X_test)
    #evaluating the model using classification_report
    for i in range(36):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i], target_names=Y_test.columns))
    pass


def save_model(model, model_filepath):
    #Saving model
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()