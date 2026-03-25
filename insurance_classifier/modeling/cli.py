import typer

# from pathlib import Path
from insurance_classifier.modeling.train import train_model

app = typer.Typer()

@app.command()
def train(
    base_config: str = typer.Option(..., help="Path to the base config .yaml file"),
    model_config: str = typer.Option(..., help="Path to the model config .yaml file"),
    model_version: str = typer.Option(..., help="Version of the model to train")

):
    train_model(base_config, model_config, model_version)

if __name__ == "__main__":
    app()
