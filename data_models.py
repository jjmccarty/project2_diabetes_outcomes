from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score

import data_outcomes


def get_train_test_split(X, y):
    """
        Ensures the train_test_split call is consistent for all regressions
    """
    return train_test_split(X, y, random_state=42, stratify=y)

def preprocess_prod_data(df):
    """
    Dropping columns not relevant to data. 
    TODO - fill in with the data we want to pre-process and options
    """

    X = df.copy()
    X = X.drop(columns=['Diabetes'])
    y = df['Diabetes']
    #totally don't remember why we need to do this sometimes.
    #y = df['Diabetes'].values.reshape(-1,1)

    return get_train_test_split(X, y)
    #return df

def preprocess_prod_data2(df):
    """
    Dropping columns not relevant to data. 
    PLACE HOLDER FOR IF WE WANT TO TEST UTILIZING DIFFERENT FEATURE SETS.
    THE CONTENT WOULD BE REPLACED BY THE CHANGES WE WANT TO UTILIZE.

    RIGHT NOW JUST RANDOMLY REMOVED FEATURES AS AN EXAMPLE!!!!!
    """
    
    X = df.copy()
    X = X.drop(columns=['Diabetes', 'DiffWalk', 'PhysHlth','MentHlth','GenHlth'])
    y = df['Diabetes']
    #y = df['Diabetes'].values.reshape(-1,1)

    return get_train_test_split(X, y)

'''
def r2_adj(x, y, model):
    """
    Calculates adjusted r-squared values given an X variable, 
    predicted y values, and the model used for the predictions.
    """
    r2 = model.score(x,y)
    n = x.shape[0]
    p = y.shape[1]
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
'''

def r2_adj(x, y, model):
    r2 = model.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)


def check_metrics(X_test, y_test, model):
    # Use the pipeline to make predictions
    y_pred = model.predict(X_test)

    # Print out the MSE, r-squared, and adjusted r-squared values
    #print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    #print(f"R-squared: {r2_score(y_test, y_pred)}")
    #print(f"Adjusted R-squared: {r2_adj(X_test, y_test, model)}")  

    return r2_adj(X_test, y_test, model) 

def get_best_pipeline(pipeline1, pipeline2, df):
    """
    Accepts two pipelines and the dataframe.
    Uses two different preprocessing functions to split the data for training 
    the different pipelines, then evaluates which pipeline performs best.
    """
    # Apply the preprocess_rent_data step
    X_train, X_test, y_train, y_test = preprocess_prod_data(df)

    # Fit the first pipeline
    pipeline1.fit(X_train, y_train)

    print("Testing all features")
    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the first pipeline
    p1_adj_r2 = check_metrics(X_test, y_test, pipeline1)

    # Apply the preprocess_data without date information
    X_train, X_test, y_train, y_test = preprocess_prod_data2(df)

    # Fit the second pipeline
    pipeline2.fit(X_train, y_train)

    print("Testing dropping features")
    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the second pipeline
    p2_adj_r2 = check_metrics(X_test, y_test, pipeline2)
    #print(p2_adj_r2)


    # Compare the adjusted r-squared for each pipeline and 
    # return the best model
    if p2_adj_r2 > p1_adj_r2:
        print("Returning reduced feature subset")
        X_train, X_test, y_train, y_test = preprocess_prod_data2(df)
        pipeline2.fit(X_train, y_train)
        pipeline = pipeline2
        return X_train, X_test, y_train, y_test, pipeline
        #return pipeline2
    else:
        print("Returning all features")
        X_train, X_test, y_train, y_test = preprocess_prod_data(df)
        pipeline1.fit(X_train, y_train)
        pipeline = pipeline1
        return X_train, X_test, y_train, y_test, pipeline
        #return pipeline    



def model_generator(df, regression_model):
    """
    TODO
    """
    # Create a list of steps for a pipeline that will one hot encode and scale data
    # Each step should be a tuple with a name and a function
    #steps = [("Scale", StandardScaler(with_mean=False)), 
    #         ("Linear Regression", LinearRegression())] 

    #steps = [("Scale", StandardScaler(with_mean=False)),regression_model] 
    steps = [('scaler', StandardScaler(with_mean=False)), ('model', regression_model)]

    # Create a pipeline object
    pipeline1 = Pipeline(steps)

    pipeline2 = Pipeline(steps)

    # Get the best pipeline
    #pipeline = get_best_pipeline(pipeline1, pipeline2, df)

    # Return the trained model
    #return pipeline
    return get_best_pipeline(pipeline1, pipeline2, df)



def get_metrics(model_name, y_test, predictions):
    d = {}
    d['model'] = model_name
    accuracy = accuracy_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions, labels=[1,0])
    classification = classification_report(y_test, predictions, labels = [1, 0])
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    d['accuracy'] = accuracy
    d['confusion_matrix'] = confusion
    d['classification_report'] = classification
    d['balanced_accuracy_score'] = balanced_accuracy
    return d


def model_selector(df, models):


    d = {}
    for m in models:

        print(f'------------ Running predictions for {m}  --------------------')
        model_data = {}
        #X_train, X_test, y_train, y_test, pipeline = model_generator(df, LogisticRegression(random_state=42))
        X_train, X_test, y_train, y_test, pipeline = model_generator(df, m)

        training_predictions = pipeline.predict(X_train)
        testing_predictions = pipeline.predict(X_test)
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        model_data['model'] = type(pipeline.named_steps['model'])
        model_data['train_score'] = train_score
        model_data['test_score'] = test_score
        test_metrics = get_metrics(model_data['model'], y_test, testing_predictions)
        model_data['test_metrics'] = test_metrics
        d[model_data['model']] = model_data

    return d

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")

    df = data_outcomes.getDiabetesBehaviorDataframe()
   # X_train, X_test, y_train, y_test, model = model_generator(df, LogisticRegression(random_state=42))


    
    models = [
        LogisticRegression(random_state=42),
        KNeighborsClassifier(n_neighbors=27),
        GradientBoostingClassifier(random_state=42),
        AdaBoostClassifier(random_state=42)
    ]
    dval = model_selector(df, models)
    print(dval)

