from pathlib import Path

import joblib
import pandas as pd


class PreprocessingPipeline:
    """Prepare incoming API payloads into the exact model feature matrix."""

    def __init__(self, config):
        self.config = config
        parameters_path = Path(config["paths"]["parameters_dir"])

        self.one_hot_encoder = joblib.load(
            parameters_path / config["artifacts"]["one_hot_encoder"]
        )
        self.policy_sales_channel_encoder = joblib.load(
            parameters_path / config["artifacts"]["policy_sales_channel_encoder"]
        )
        self.age_scaler = joblib.load(parameters_path / config["artifacts"]["age_scaler"])
        self.annual_premium_scaler = joblib.load(
            parameters_path / config["artifacts"]["annual_premium_scaler"]
        )
        self.vintage_scaler = joblib.load(parameters_path / config["artifacts"]["vintage_scaler"])
        self.region_code_encoder = joblib.load(
            parameters_path / config["artifacts"]["region_code_encoder"]
        )

    @staticmethod
    def _normalize_text(value):
        if value is None:
            return None
        return str(value).strip().lower()

    @staticmethod
    def _to_snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result.columns = [col.strip().lower().replace(" ", "_") for col in result.columns]
        return result

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._to_snake_case_columns(df)

        vehicle_age_mapping = {
            "> 2 years": "over_2_years",
            "1-2 year": "1_to_2_years",
            "< 1 year": "under_1_year",
            "over_2_years": "over_2_years",
            "1_to_2_years": "1_to_2_years",
            "under_1_year": "under_1_year",
        }

        categorical_columns = ["gender", "vehicle_age", "vehicle_damage"]
        for column in categorical_columns:
            if column in df.columns:
                df[column] = df[column].map(self._normalize_text)

        if "vehicle_age" in df.columns:
            df["vehicle_age"] = df["vehicle_age"].map(vehicle_age_mapping)

        return df

    def _validate_input_columns(self, df: pd.DataFrame) -> None:
        required_columns = [
            "gender",
            "age",
            "driving_license",
            "region_code",
            "previously_insured",
            "vehicle_age",
            "vehicle_damage",
            "annual_premium",
            "policy_sales_channel",
            "vintage",
        ]
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Missing required columns: {missing}")

    def data_preparation(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input_columns(df)
        data = df.copy()

        data["vehicle_damage"] = data["vehicle_damage"].map({"yes": 1, "no": 0, "1": 1, "0": 0})

        one_hot_features = self.one_hot_encoder.transform(data[["gender", "vehicle_age"]])
        one_hot_df = pd.DataFrame(
            one_hot_features,
            columns=self.one_hot_encoder.get_feature_names_out(["gender", "vehicle_age"]),
            index=data.index,
        )

        data["region_code"] = self.region_code_encoder.transform(data[["region_code"]]).iloc[:, 0]

        frequency_encoder_features = list(
            getattr(self.policy_sales_channel_encoder, "feature_names_in_", [])
        )
        if not frequency_encoder_features:
            frequency_encoder_features = [
                "gender",
                "age",
                "driving_license",
                "region_code",
                "previously_insured",
                "vehicle_age",
                "vehicle_damage",
                "annual_premium",
                "policy_sales_channel",
                "vintage",
            ]

        freq_encoded = self.policy_sales_channel_encoder.transform(
            data[frequency_encoder_features]
        )
        data["policy_sales_channel"] = freq_encoded["policy_sales_channel"]

        data["age"] = self.age_scaler.transform(data[["age"]].values).ravel()
        data["vintage"] = self.vintage_scaler.transform(data[["vintage"]].values).ravel()
        data["annual_premium"] = self.annual_premium_scaler.transform(
            data[["annual_premium"]].values
        ).ravel()

        final_df = pd.concat([data, one_hot_df], axis=1)
        expected_features = [
            "age",
            "driving_license",
            "region_code",
            "previously_insured",
            "vehicle_damage",
            "annual_premium",
            "policy_sales_channel",
            "vintage",
            "gender_female",
            "gender_male",
            "vehicle_age_1_to_2_years",
            "vehicle_age_over_2_years",
            "vehicle_age_under_1_year",
        ]

        return final_df[expected_features]
