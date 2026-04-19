import os
from pathlib import Path
from datetime import datetime
import json
from src.tools.schemas import PredictionReport


def create_prediction_report(
    save_name: str,
    str_uuid: str,
    prediction: list[tuple[int, float, int]],
    features_list: list,
    metadata_name: str,
    allow_missing_features: bool,
    threshold: float,
    save_dir: str,
    save_result: bool,
) -> PredictionReport:
    os.makedirs(save_dir, exist_ok=True)

    report = {
        "metadata": {
            "uuid": str_uuid,
            "metadata_name": metadata_name,
            "allow_missing_features": allow_missing_features,
            "threshold": threshold,
            "features_list": features_list,
            "timestamp": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
        },
        "predictions": [
            {"data_id": id, "prediction": pred, "probability": round(proba, 3)}
            for id, proba, pred in prediction
        ],
    }

    if save_result:
        with open(Path(save_dir) / save_name, "w") as f:
            json.dump(report, f, indent=4)
    return PredictionReport(**report)
