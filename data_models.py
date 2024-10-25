"""
    The data_models function in this python script are designed to 
    encapsulate the ability to test preprocessor options/refinements and 
    a variety of models against a consistent train test data set to 
    minimize repeat code in the notebooks.  The intent is to streamline 
    comparison and selection of a model for future production implementation. 


    1.  Initial calls to model_selector(df, models) will start the process.  
        Mutliple models can be passed in the array, but it is assumed that 
        the models are refined.
    2.  The model_selector will call the get_best_pipeline function which
        creates the pipeline including necessary categorization and scaling
        processes. 
    3. The model_selector will compare preprocessor.  To do this 2 functions
        are provided preprocess_prod_data and preprocess_prod_data2.  
        - preprocess_prod_data should have the current most effective 
          preprocessing routine of the feature data.  
        - preprocess_prod_data2 can be modified to provide a modification to 
          the current preprocessor to evaluate if it is more or less 
          effective.  The most effective processor is determined by r2 adj.
    4.  All routines utilize the get_train_test_split function to ensure that 
        the train/test data is consistently applied for all models

"""


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
        Ensures the train_test_split call is consistent for all regressions.
        Data is broken into a standard (75/25) split with a random state of 42
        to ensure consistency in the training and test data.  
        Because the data is imbalanced stratify is used to ensure that that 
        each set of data has the same proportion for the target in the set. 

        args:
            X - the data frame holding the feature set data
            y - the data for the target outcome

        returns:
            The train test split options
    """
    return train_test_split(X, y, random_state=42, stratify=y)

def preprocess_prod_data(df):
    """
    Default preprocessing for the data set.  This is the current standard
    against to test feature manipulation, prior to the train/test split.  

    This will call the get_train_test_split function to ensure all 
    preprocessors are using the same training and test data.
    
    args:
        df - the dataframe to manipulate

    returns:
        The train test split data
    """

    X = df.copy()
    X = X.drop(columns=['Diabetes','Stroke', 'HeartDiseaseorAttack','MentHlth', 'DiffWalk'])
    y = df['Diabetes']

    return get_train_test_split(X, y)

def preprocess_prod_data2(df):
    """
    Testing of alternate preprocessor data.  This will perform the same 
    routine as the preprocess_prod_data function but allows for alternative
    manipulation and processing of data.
    
    args:
        df - the dataframe to manipulate

    returns:
        The train test split data
    """
    
    X = df.copy()
    X = X.drop(columns=['Diabetes', 
                        'Stroke', 
                        'HeartDiseaseorAttack',
                        'GenHlth', 
                        'MentHlth', 
                        'PhysHlth', 
                        'DiffWalk'])
    y = df['Diabetes']

    return get_train_test_split(X, y)

def r2_adj(x, y, model):
    """
    Utility to calculate the adjusted r2 value
    
    args:
        x - feature data
        y - target data
        model - model used for scoring

    returns:
        The adjusted r2 value
    """
    r2 = model.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)


def check_metrics(X_test, y_test, model):
    """
     Retrieves the adjusted r2 value for the model
     args:
        x - feature data
        y - target data
        model - model used for scoring

    returns:
        The adjusted r2 value
    """
    y_pred = model.predict(X_test)
    print(f"Adjusted R-squared: {r2_adj(X_test, y_test, model)}")  
    return r2_adj(X_test, y_test, model) 

def get_best_pipeline(pipeline1, pipeline2, df):
    """
    Note: Adapted from prior class mini-project in the SMU AI Bootcamp
    Accepts two preprocessor pipelines and the dataframe.
    Uses two different preprocessing functions to split the data for training 
    the different pipelines, then evaluates which pipeline performs best.
    args:
        pipeline1 - the first pipeline to process
        pipeline2 - the second pipeline to process
        df - the data frame to process
    returns:
        The best preprocessor pipeline and the train/test split data as
        X_train, X_test, y_train, y_test, model
    """
    # Apply the preprocess_rent_data step
    X_train, X_test, y_train, y_test = preprocess_prod_data(df)

    # Fit the first pipeline
    pipeline1.fit(X_train, y_train)

    print("Testing all features")
    # collect the R2 value
    p1_adj_r2 = check_metrics(X_test, y_test, pipeline1)

    # Apply the preprocess_data without date information
    X_train, X_test, y_train, y_test = preprocess_prod_data2(df)

    # Fit the second pipeline
    pipeline2.fit(X_train, y_train)

    print("Testing dropping features")
    # collect the R2 value
    p2_adj_r2 = check_metrics(X_test, y_test, pipeline2)

    # Compare the adjusted r-squared for each pipeline and 
    # return the best model
    if p2_adj_r2 > p1_adj_r2:
        print("Returning reduced feature subset")
        X_train, X_test, y_train, y_test = preprocess_prod_data2(df)
        pipeline2.fit(X_train, y_train)
        pipeline = pipeline2
        return X_train, X_test, y_train, y_test, pipeline
    else:
        print("Returning all features")
        X_train, X_test, y_train, y_test = preprocess_prod_data(df)
        pipeline1.fit(X_train, y_train)
        pipeline = pipeline1
        return X_train, X_test, y_train, y_test, pipeline
   



def model_generator(df, regression_model):
    """
    Note: Adapted from prior class mini-project in the SMU AI Bootcamp
    Creates 2 pipelines for a specific regression and executes the 
    get_best_pipeline function to determine and return the pipeline with the
    best pre-processor.  
    args:
        df - the working data frame
        regression_model - the model to execute in the pipleine.
    returns:
        the result of the get_best_pipeline call per that function
    """
    # Create a list of steps for a pipeline that will one hot encode and scale data
    steps = [('scaler', StandardScaler(with_mean=False)), ('model', regression_model)]

    # Create a pipeline object
    pipeline1 = Pipeline(steps)
    pipeline2 = Pipeline(steps)
    return get_best_pipeline(pipeline1, pipeline2, df)



def get_metrics(model_name, y_test, predictions):
    """
    Processes all standard metrics for a given model and returns the data as
    a dictionary
    args:
        model_name - reference to the model name being executed
        y_test - y_test data
        predictions - the prediction set for the model
    returns:
        dictionary of the model metrics
    """
    d = {}
    d['model'] = model_name
    accuracy = accuracy_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions, labels=[1,0])
    classification = classification_report(y_test, predictions, labels = [1, 0],output_dict=True, digits=5)
    
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    d['accuracy'] = accuracy
    d['confusion_matrix'] = confusion
    d['classification_report'] = classification
    d['balanced_accuracy_score'] = balanced_accuracy
    return d


def model_selector(df, models):
    """
    Runs the model predictions for an array of model objects for the purposes
    of refining and comparing the results of multiple models against a 
    standard dataset.  It is assumed that models passed include all 
    necessary refinements such as hyperparameter tuning. 

    The standard set of metrics for the models are returned as a dictionary of
    values. 

    The primary purpose of this function is to allow for easy comparison of 
    multiple models for selection.  

    args:
        df - the data frame to utilize with the models.  Note this will be 
        run through the get_train_test_split to ensure a consist execution
        against the training and testing data for all models. 
        modesl - an array of model/regression objects (including tunings) to
        execute predictions on.  

    returns:
        a dictionary of metrics for the model for comparison purposes. 
    """

    d = {}
    for m in models:

        print(f'------------ Running predictions for {m}  --------------------')
        model_data = {}
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


def retrieve_model(df, model):
    print(f'------------ Running predictions for {model}  --------------------')
    X_train, X_test, y_train, y_test, pipeline = model_generator(df, model)
    return X_train, X_test, y_train, y_test, pipeline

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")

    df = data_outcomes.getDiabetesBehaviorDataframe()
    X_train, X_test, y_train, y_test, pipeline = retrieve_model(df, GradientBoostingClassifier(random_state=42, n_estimators=200))