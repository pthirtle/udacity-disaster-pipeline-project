import sys
from sqlalchemy import create_engine
import pandas as pd
import re

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(database_filepath):
    """
    Load Data Function

    Parameters
    ----------
    database_filepath : string -> location of database

    Returns
    -------
    X -> text from messages
    y -> multiple labels  for each message (1 or 0)
    column_names -> column headings for y

    """  
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('labelled_messages', engine)
    # text is stored in the message column
    X = df.message.values
    # categories are in columns 4 onwards
    y = df.iloc[:,4:]
    # get category names
    category_names = y.columns

    return X, y, category_names

def tokenize(text):
    """
    Tokenize Function

    Parameters
    ----------
    text : string -> test string to tokenize
    
    Returns
    -------
    clean_tokens : list -> tokens

    """
    # replace urls with placeholder
    # regex to select urls 
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # tokenize text
    tokens = word_tokenize(text)    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


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




def build_model():
    """
    Build Model Function
    
    Builds a model pipeline including GridSearch

    Returns
    -------
    cv_model : model -> the model 

    """
    # create the pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),
            ('starting_verb_transformer', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))    
    ])
    
    # set the parameters for grid search
    parameters = [{ 'clf__estimator__n_estimators': [5, 10, 20]
#                   ,'clf__estimator__max_depth': [5, 10, None],
#                   ,'clf__estimator__max_leaf_nodes': [5, 10, None]
                   }]
    cv_model =  GridSearchCV(pipeline, param_grid=parameters, verbose = 1)
    
    return cv_model
    


def evaluate_model(model, X_test, Y_test, category_names):
    
    # get classification report
    y_pred = model.predict(X_test)
    for category in category_names:
        print('Model Performance with Category: {}'.format(category))
        print(classification_report(Y_test[category],y_pred[category]))
        print("\nBest Parameters:", model.best_params_)

def save_model(model, model_filepath):
    
    # save model
    pickle.dump(model, open(model_filepath,'wb'))


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