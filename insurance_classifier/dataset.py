import joblib
import pandas as pd

# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = RAW_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     # ----------------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Processing dataset...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Processing dataset complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()


class DataPreprocessor(object):

    def __init__(self):
        self.age_scaler =                   joblib.load(open('../parameters/age_scaler.joblib', 'rb')) 
        self.annual_premium_scaler =        joblib.load(open('../parameters/annual_premium_scaler.joblib', 'rb'))
        self.vintage_scaler =               joblib.load(open('../parameters/vintage_scaler.joblib', 'rb'))
        self.region_code_encoder =          joblib.load(open('../parameters/region_code_james_stein.joblib', 'rb'))               
        
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
        df = pd.get_dummies(df, columns= ['gender'], prefix= 'gender', dtype=int)

        # Target encoding (James-Stein) 
        df['region_code'] = self.region_code_encoder.transform(X= df[['region_code']])

        # Label encoding
        df['vehicle_damage'] = df['vehicle_damage'].map({'yes':1,'no':0})

        # One-Hot encoding
        df = pd.get_dummies(df, columns= ['vehicle_age'], prefix= 'vehicle_age', dtype=int)

        # Frequency encoding
        fe_policy_sales_channel = df['policy_sales_channel'].value_counts(normalize=True)
        df['policy_sales_channel'] = df['policy_sales_channel'].map(fe_policy_sales_channel)

        # Rescaling -----
        df['age'] = self.age_scaler.fit_transform(df[['age']].values)
        df['vintage'] = self.vintage_scaler.fit_transform(df[['vintage']].values)

        # Standardization -----
        df['annual_premium'] = self.annual_premium_scaler.fit_transform(df[['annual_premium']].values)

        # Feature selection
        features_selected = ['vintage',
                  'annual_premium',
                  'age',
                  'region_code',
                  'vehicle_damage',
                  'policy_sales_channel',
                  'previously_insured']

        return df[features_selected]
    
    
    

