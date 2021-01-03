#Import libraries 

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, recall_score, plot_confusion_matrix, classification_report, f1_score, accuracy_score

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')


def transform_split_data(df, target: str, scale: bool):
    
    """Inputs: 
    df : DataFrame = data as data frame
    target : str = name of target variable
    scale : bool = specify if you want to scale the data using a StandardScaler()
    
    
    Retruns: X_train, X_test, y_train, y_test
    
    This function performs several forms of pre-processing: dummifies all categorical
    features in the dataframe, splits your data into train and test sets, scales the data
    using StandardScaler() and, finally, uses SMOTE to fix class imbalance in your data.
    It returns proprely processed sets of your training and testing data.
   
    ------------------------
    """ 
    
    #Dummify Categorical Variables
    df = pd.get_dummies(df, drop_first=True)

    # Split the data into train and test sets
    X = df.drop([target], axis=1)
    y = df[target]
    x_columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    
    
    if scale == True: 
        #Scaling with StandardScaler
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        scaled_X_train = pd.DataFrame(X_train_scaled, columns = x_columns)
        scaled_X_test = pd.DataFrame(X_test_scaled, columns = x_columns)

        os = SMOTE(random_state=0)
        columns = scaled_X_train.columns

        os_data_X,os_data_y=os.fit_sample(scaled_X_train, y_train)
        os_data_X = pd.DataFrame(data=os_data_X)
        os_data_y= pd.Series(data=os_data_y)
        
        return os_data_X, scaled_X_test, os_data_y, y_test
    
    else: 
        
        os = SMOTE(random_state=0)
        columns = X_train.columns

        os_data_X,os_data_y=os.fit_sample(X_train, y_train)
        os_data_X = pd.DataFrame(data=os_data_X)
        os_data_y= pd.Series(data=os_data_y)
        
        return os_data_X, X_test, os_data_y, y_test




def compare_models(X_tr, X_tst, y_tr, y_tst, models, names):
       
        """Inputs: 
        X_tr = your training data as pd.DataFrame
        X_tst = your testing data as pd.DataFrame
        y_tr = your training target as pd.Series
        y_tst = your testing target as pd.Series
        models = a list of model objects you want to compare
        names = a list of strings containing the names of your modelds 


        Retruns: a comparison table inclduing Recall, Accuracy and F1 Score for each.
       
        ------------------------
        """ 
    
    
        X_train, X_test, y_train, y_test = X_tr, X_tst, y_tr, y_tst
       
        recall_results = []
        accuracy_results = []
        f1_scores = []
        
        for i in range(len(models)):
            clf = models[i].fit(X_train, y_train)
            
            print("\n")
            print('-'*100)
            print(names[i])
            cmatrix = plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.BuPu, display_labels = ['Retain', 'Churn'])
            plt.show()
            
            recall = recall_score(y_test, clf.predict(X_test))
            recall_results.append(recall)
            accuracy = accuracy_score(y_test, clf.predict(X_test))
            accuracy_results.append(accuracy)
            f1 = f1_score(y_test, clf.predict(X_test))
            f1_scores.append(f1)
            
            print("Here's how the {} model performed with the TRAINING data: \n".format(names[i]))
            print(classification_report(y_train, clf.predict(X_train)))
            print('-'*75)
            print("Here's how the {} model performed with the TESTING data: \n".format(names[i]))
            print(classification_report(y_test, clf.predict(X_test)))
                        
            
        col1 = pd.DataFrame(names)
        col2 = pd.DataFrame(recall_results)
        col3 = pd.DataFrame(accuracy_results)
        col4 = pd.DataFrame(f1_scores)

        results = pd.concat([col1, col2, col3, col4], axis='columns')
        results.columns = ['Model', 'Recall Score', "Accuracy(Test)", "F1 Score",]
        
        return results