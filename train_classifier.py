import sys
import pandas as pd
from sqlalchemy import create_engine

# libraries for NLP
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#libraries for Modeling
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import pickle


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """Load the database data and return a dataframe.
    
    Input
    database_filepath: str. the database location.
    
    Outputs
    X: Series containing the text to be used in model
    Y: Dataframe containing target variables for the model
    categories_names: list of target variable names. 
    """
    
    # load the data
    engine = create_engine('sqlite:///DisasterTextTable.db')
    df = pd.read_sql_table(database_filepath,engine)
    
    # split df into X, Y variables, and extract category names
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = Y.columns 
    
    return X, Y, category_names


def tokenize(text):
    """Convert the text array into a list of tokens.
    
    Input
    text: array of texts to be used for the model.
    
    Output
    tokens: list of tokens extracted from text.
    
    """
    
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens



def build_model():
    """ Build a Classification ML pipeline 
    
    Output
    pipeline: classification model.
    """
    
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('multi',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline    



def evaluate_model(model, X_test, Y_test, category_names):
    """ Takes the fit model and evaluates the model given the X and Y tests.
    
    Inputs
    model: obj. pipeline classification model 
    X_test: 
    Y_test: 
    category_names: list
    
    Output
    print classification report
    """
    
    # use model to predict X_test 
    y_pred = model.predict(X_test)
    
    #print the classification_report
    print(classification_report(Y_test,y_pred,target_names=category_names))

def save_model(model, model_filepath):
    """ Save the model to a pickle file
    
    Inputs
    model: obj. pipeline containing classification model
    model_filepath: str. the location where the model will get saved
    
    Output
    pickle file
    """
    
    pickle.dump(model,open(model_filepath,'wb'))



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