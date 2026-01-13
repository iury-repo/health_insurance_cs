import typer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from insurance_classifier.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


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


def DataPreprocessing(df):

    X_df = df.drop(['id','response'], axis=1)
    y_df = df['response'].copy()

    # Rename vehicle_age categorie
    X_df['vehicle_age'] = X_df['vehicle_age'].map({'> 2 Years': 'over_2_years',
                                                        '1-2 Year': '1_to_2_years',
                                                        '< 1 Year': 'under_1_year'
                                                        })

    # Change string entries to snakecase
    for col in X_df.select_dtypes(exclude=['int64','float64', 'datetime64[ns]']).columns:
        X_df[col] = X_df[col].str.lower()

    # Encoding -----

    # One-Hot encoding
    X_df = pd.get_dummies(X_df, columns= ['gender'], prefix= 'gender', dtype=int)

    # Target encoding (James-Stein) 
    tg_enc_region_code = ce.JamesSteinEncoder(cols=['region_code'])
    X_df['region_code'] = tg_enc_region_code.fit_transform(X= X_df[['region_code']], y= y_df)

    # Label encoding
    X_df['vehicle_damage'] = X_df['vehicle_damage'].map({'yes':1,'no':0})

    # One-Hot encoding
    X_df = pd.get_dummies(X_df, columns= ['vehicle_age'], prefix= 'vehicle_age', dtype=int)

    # Frequency encoding
    fe_policy_sales_channel = X_df['policy_sales_channel'].value_counts(normalize=True)
    X_df['policy_sales_channel'] = X_df['policy_sales_channel'].map(fe_policy_sales_channel)

    # Rescaling -----

    mms_age = MinMaxScaler()
    mms_vintage = MinMaxScaler()

    X_df['age'] = mms_age.fit_transform(X_df[['age']].values)
    X_df['vintage'] = mms_age.fit_transform(X_df[['vintage']].values)

    # Standardization -----

    # annual_premium
    ss = StandardScaler()
    X_df['annual_premium'] = ss.fit_transform(X_df[['annual_premium']].values)

    return df
