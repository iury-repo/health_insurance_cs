from xgboost import XGBClassifier

MODEL_REGISTRY = {
    "xgboost": XGBClassifier
}

def build_model(config):
    model_type = config['model']['type']

    params = config['model']['params']
    model_class = MODEL_REGISTRY[model_type]

    return model_class(**params)

