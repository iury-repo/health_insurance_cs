from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

def feature_engineering(df):
    df = df.copy()

    df.columns = [col.lower().replace(' ','_') for col in df.columns]

    df['vehicle_age'] = df['vehicle_age'].map({
        '> 2 Years': 'over_2_years',
        '1-2 Year': '1_to_2_years',
        '< 1 Year': 'under_1_year'
    })

    for col in df.select_dtypes(exclude=['int64','float64']):
        df[col] = df[col].astype(str).str.lower()

    return df