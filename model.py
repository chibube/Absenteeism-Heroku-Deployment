import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self,columns=None,copy=True, with_mean=True, with_std=True):
        self.Scaler = StandardScaler()
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

class absenteeism_model():
    
    def __init__(self, model_file, scaler_file):
        # reading the 'model' and 'scaler' files which were saved
        # we need to pass the CustomScaler class into __main__
        if __name__ = '__main__':
            CustomScaler()
            with open('model','rb') as model_file, open('scaler','rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
            
    #automate the preprocessing of the '.csv' file
    def load_and_clean_data(self, data_file):
                
        #import the data and storing in different variables for separate use
        df = data_file
        self.df_with_predictions = df.copy()
        df = df.drop(['ID'], axis = 1)
        df['Absenteeism Time in Hours'] = 'NaN'
        
        #creating df containing dummy variables
        #reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
        #since we will be taking in inputs one at a time we might not have as many variables
        #to get the number of dummy variables as in the training so we will use an if stm

        if df['Reason for Absence'][0] <= 14:
             #split reason_columns into 4 types
            reason_type_1 = pd.Series(1)
            reason_type_2 = pd.Series(0)
            reason_type_3 = pd.Series(0)
            reason_type_4 = pd.Series(0)

        elif df['Reason for Absence'][0] in range (15, 18):
            reason_type_1 = pd.Series(0)
            reason_type_2 = pd.Series(1)
            reason_type_3 = pd.Series(0)
            reason_type_4 = pd.Series(0)
        elif df['Reason for Absence'][0] in range (18, 22):
            reason_type_1 = pd.Series(0)
            reason_type_2 = pd.Series(0)
            reason_type_3 = pd.Series(1)
            reason_type_4 = pd.Series(0)
        else:
            reason_type_1 = pd.Series(0)
            reason_type_2 = pd.Series(0)
            reason_type_3 = pd.Series(0)
            reason_type_4 = pd.Series(1)

        #to avoid multicollinearity, we will drop the "REason for absence column" and replace with the dummies
        df = df.drop(['Reason for Absence'], axis = 1)
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)
       
        #naming the columns and rearranging
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                           'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                           'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
        df.columns = column_names
        column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 
                                      'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education', 
                                      'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]
        
        #converting date column into datetime format and splitting into months & days of the week
        df['Date'] = pd.to_datetime(df['Date'])
        list_months = []
        for i in range(df.shape[0]):
            list_months.append(df['Date'][i].month)
            
        df['Month Value'] = list_months
        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())
        
        #There is no need for the date column any more
        df = df.drop(['Date'], axis = 1)
        
        column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',
                                'Transportation Expense', 'Distance to Work', 'Age',
                                'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                                'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_upd]
        
        #mapping education variables to separate graduate+ qualified employees from undergrad employees
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        
        df = df.fillna(value=0)
        
        df = df.drop(['Absenteeism Time in Hours'], axis=1)
        
        df = df.drop(['Day of the Week', 'Daily Work Load Average', 'Distance to Work'], axis=1)
        
        self.preprocessed_data = df.copy()
        
        #This will enable us in the next function
        self.data = self.scaler.transform(df)
        
    #this function outputs the probability of a data point to be 1
    def predicted_probability(self):
        if(self.data is not None):
            pred = self.reg.predict_proba(self.data)[:, 1]
            return pred 
    
    #outputs 0s and 1s based on our model
    def predicted_output_category(self):
        if(self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
        
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data
