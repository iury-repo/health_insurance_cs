from pathlib import Path

import joblib
import pandas as pd

from insurance_classifier.config import load_yaml_config

config = load_yaml_config('config/base.yaml')

class PreprocessingPipeline(object):

    def __init__(self, config):
        self.config = config
        parameters_path = Path(config['paths']['parameters_dir'])

        self.one_hot_encoder =              joblib.load(parameters_path / config['artifacts']['one_hot_encoder'])
        self.policy_sales_channel_encoder = joblib.load(parameters_path / config['artifacts']['policy_sales_channel_encoder'])  
        self.age_scaler =                   joblib.load(parameters_path / config['artifacts']['age_scaler']) 
        self.annual_premium_scaler =        joblib.load(parameters_path / config['artifacts']['annual_premium_scaler'])
        self.vintage_scaler =               joblib.load(parameters_path / config['artifacts']['vintage_scaler'])
        self.region_code_encoder =          joblib.load(parameters_path / config['artifacts']['region_code_encoder'])               
        
    def feature_engineering(self, df):  
          
        # Change column names to snakecase
        df.columns = [col.lower().replace(' ','_') for col in df.columns]

        # Rename vehicle_age categories
        df['vehicle_age'] = df['vehicle_age'].map({'> 2 Years': 'over_2_years',
                                                            '1-2 Year': '1_to_2_years',
                                                            '< 1 Year': 'under_1_year'
                                                            })

        # Change string entries to snakecase
        for col in df.select_dtypes(exclude=['int64','float64', 'datetime64[ns]']).columns:
            df[col] = df[col].str.lower()

        return df

    def data_preparation(self, df):
        # One-Hot encoding
        df['gender'] = self.one_hot_encoder.transform(df[['gender']])
        df['vehicle_age'] = self.one_hot_encoder.transform(df[['vehicle_age']])
        # df[['gender', 'vehicle_age']] = self.one_hot_encoder.transform(df[['gender', 'vehicle_age']])

        # Target encoding (James-Stein) 
        df['region_code'] = self.region_code_encoder.transform(X= df[['region_code']])

        # Label encoding
        df['vehicle_damage'] = df['vehicle_damage'].map({'yes':1,'no':0})

        # Frequency encoding
        df['policy_sales_channel'] = self.policy_sales_channel_encoder.transform(df[['policy_sales_channel']])

        # Rescaling -----
        df['age'] = self.age_scaler.transform(df[['age']].values)
        df['vintage'] = self.vintage_scaler.transform(df[['vintage']].values)

        # Standardization -----
        df['annual_premium'] = self.annual_premium_scaler.transform(df[['annual_premium']].values)

        return df[self.config['features']['selected_features']]
    
    
    

