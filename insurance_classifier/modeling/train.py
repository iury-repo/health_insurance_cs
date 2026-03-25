from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from insurance_classifier.config import load_yaml_config
from insurance_classifier.modeling.build_model import build_model


def train_model(base_config: str, model_config: str, model_version: str):
    # Load config files
    b_config = load_yaml_config(base_config)
    m_config = load_yaml_config(model_config)

    # Load training data
    try:
        X_train = pd.read_csv(b_config['paths']['processed_data'] + '/X_train.csv')
    except Exception as e:
        logger.error(e)
        raise
    try:
        y_train = pd.read_csv(b_config['paths']['processed_data'] + '/y_train.csv').values.ravel()
    except Exception as e:
        logger.error(e)
        raise   

    # Build model
    model = build_model(m_config)

    # Train model
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), transient=True,) as progress:
        task = progress.add_task("Training model...", total=None)

        model.fit(X_train, y_train)

        progress.remove_task(task)

    # Save model
    model_dir = Path(b_config['paths']['model_dir'])
    model_path = model_dir / f"{m_config['model']['type']}_{model_version}.joblib"

    joblib.dump(model, model_path)

    logger.success(f"Training finished. Model saved to {model_path}")