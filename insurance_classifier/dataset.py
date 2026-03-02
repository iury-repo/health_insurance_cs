from pathlib import Path

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

    def __init__(self, config):
        self.config = config
        base_path = Path(config['paths']['parameters_dir'])

        self.age_scaler =                   joblib.load(base_path / config['artifacts']['age_scaler']) 
        self.annual_premium_scaler =        joblib.load(base_path / config['artifacts']['annual_premium_scaler'])
        self.vintage_scaler =               joblib.load(base_path / config['artifacts']['vintage_scaler'])
        self.region_code_encoder =          joblib.load(base_path / config['artifacts']['region_code_encoder'])               
        
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
        df['age'] = self.age_scaler.transform(df[['age']].values)
        df['vintage'] = self.vintage_scaler.transform(df[['vintage']].values)

        # Standardization -----
        df['annual_premium'] = self.annual_premium_scaler.transform(df[['annual_premium']].values)

        return df[self.config['features']['selected_features']]
    
    
    

