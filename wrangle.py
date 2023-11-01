import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# sklearn:
from sklearn.preprocessing import MinMaxScaler

#new imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

import os
# SQL LOGIN
host = os.getenv('sqlHOST')
username = os.getenv('sqlUSER')
password = os.getenv('sqlPSWD')

def get_db_url(telco):
    """
    This function will:
    - take username, pswd, host credentials from imported env module
    - output a formatted connection_url
    """
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'


def new_telco_data(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a connection_url to mySQL
    - return a df of the given query from the telco_db
    """
    url = get_db_url('telco_churn')
    
    return pd.read_sql(SQL_query, url)


def get_telco_data(SQL_query, directory, filename = 'telco.csv'):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - if csv doesn't exist:
        - creates df of sql query
        - writes df to csv
    - outputs telco df
    """
    if os.path.exists(directory+filename): 
        df = pd.read_csv(filename)
        return df
    else:
        df = new_telco_data(SQL_query)

        df.to_csv(filename)
        return df
    
# telco_query = """
#         SELECT * FROM customers
#         JOIN contract_types USING (contract_type_id)
#         JOIN internet_service_types USING (internet_service_type_id)
#         JOIN payment_types USING (payment_type_id)
#         """

# telco_df = get_telco_data(telco_query, directory)
# telco_df.head()


def prep_telco_data(telco) -> pd.DataFrame:
    '''
    prep_telco will take in a a single pandas dataframe
    presumed of the same structure as presented from 
    the acquire module's get_telco_data function (refer to acquire docs)
    returns a single pandas dataframe with clean data to build and test
    our maching learning models.
    '''
    telco = telco.drop(
    columns=[
        'Unnamed: 0', 
        'payment_type_id', 
        'internet_service_type_id',
        'contract_type_id', 
        'customer_id', 
        'gender', 
        'senior_citizen',
        'partner', 
        'dependents', 
        'tenure', 
        'phone_service', 
        'multiple_lines',
        'online_security', 
        'online_backup', 
        'device_protection', 
        'tech_support',
        'streaming_tv', 
        'streaming_movies', 
        'paperless_billing',
        'monthly_charges', 
        'total_charges',])
    # Convert to correct datatypes
    # telco['contract_type'] = telco.contract_type.astype()
    
    # Get dummies for non-binary categorical variables
    dummy_telco = pd.get_dummies(telco[['contract_type', 'internet_service_type', 'payment_type']])
    telco = dummy_telco.merge(telco, left_index = True, right_index = True)
    # Concatenate dummy dataframe to original 
    # telco = pd.concat([telco, dummy_telco], axis=1)
    return telco      


def split_telco_data(telco, dataset=None):
    target_cols = {
        'telco': 'churn',
        'titanic': 'survived',
        'iris': 'species'
    }
    if dataset:
        if dataset not in target_cols.keys():
            print('please choose a real dataset tho')

        else:
            target = target_cols[dataset]
            train_val, test = train_test_split(
                df,
                train_size=0.8,
                stratify=df[target],
                random_state=1349)
            train, val = train_test_split(
                train_val,
                train_size=0.7,
                stratify=train_val[target],
                random_state=1349)
            return train, val, test
    else:
        print('please specify what df we are splitting.')
        
        
def print_cm_metrics(cm):
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn)/(tn + fp + fn + tp)

    true_positive_rate = tp/(tp + fn)
    false_positive_rate = fp/(fp + tn)
    true_negative_rate = tn/(tn + fp)
    false_negative_rate = fn/(fn + tp)

    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1_score = 2*(precision*recall)/(precision+recall)

    support_pos = tp + fn
    support_neg = fp + tn

    dict = {
        'metric' : ['accuracy'
                    ,'true_positive_rate'
                    ,'false_positive_rate'
                    ,'true_negative_rate'
                    ,'false_negative_rate'
                    ,'precision'
                    ,'recall'
                    ,'f1_score'
                    ,'support_pos'
                    ,'support_neg']
        ,'score' : [accuracy
                    ,true_positive_rate
                    ,false_positive_rate
                    ,true_negative_rate
                    ,false_negative_rate
                    ,precision
                    ,recall
                    ,f1_score
                    ,support_pos
                    ,support_neg]
    }

    return pd.DataFrame(dict)


def knn_fit_predict(k, X_train, y_train, X_validate):
    '''
    This function takes n_neighbors, X_train,  target  and X_val
    and returns knn, predictions for train set and validate set
    '''
    # MAKE the thing
    knn = KNeighborsClassifier(n_neighbors=k)

    # FIT the thing
    knn.fit(X_train, y_train)

    # USE the thing
    y_train_pred = knn.predict(X_train)
    y_validate_pred = knn.predict(X_validate)
    
    return knn, y_train_pred, y_validate_pred


def evaluate_clf(model, X, y, y_pred):
    '''
    This function can be used on any classification model
    It takes in a model, features, target and prediction
    and returns the accuracy, confusion matrix and classification report
    '''
    # model score
    accuracy = model.score(X, y)

    # confusion matrix
    cm = confusion_matrix(y, y_pred)
    cmdf = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], 
                       columns=['Pred 0', 'Pred 1'])

    # classification report
    crdf = pd.DataFrame(classification_report(y, y_pred, output_dict=True))
    
    # confusion matrix metrics
    metrics = print_cm_metrics(cm)
    
    return accuracy, cmdf, crdf, metrics