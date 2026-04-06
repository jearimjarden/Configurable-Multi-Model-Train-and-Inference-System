import pickle
from sklearn.pipeline import Pipeline
import os
from pathlib import Path
from datetime import datetime
import json


def select_best_model(report, selection_metrics):
    best = max(report, key=lambda x: report[x]["test_{}".format(selection_metrics)])
    return best


def create_artifact(
    fitted_model: dict[str, Pipeline],
    only_best: bool,
    report: dict,
    selection_metrics: str,
    save_dir: str,
):
    path = Path(save_dir)
    os.makedirs(path, exist_ok=True)
    best_model = select_best_model(report, selection_metrics)
    if only_best:
        with open(path / "best_{}.pkl".format(best_model), "wb") as f:
            pickle.dump(fitted_model[best_model], f)

    elif not only_best:
        for name, pipeline in fitted_model.items():
            with open(path / "{}.pkl".format(name), "wb") as f:
                pickle.dump(pipeline, f)


def create_metadata(
    save_dir,
    only_best: bool,
    report: dict,
    train_data,
    n_samples,
    stratify,
    target_col,
    features_col,
    random_seed,
    models,
    selection_metrics,
):
    if only_best:
        best_model = select_best_model(report, selection_metrics)
        metadata_report = {
            "run": {
                "artifact_name": best_model + ".pkl",
                "timestamp": datetime.now().strftime("%d/%m%Y, %H:%M%S"),
            },
            "model": {
                "type": models[best_model].type.name,
                "params": models[best_model].params,
            },
            "training": {
                "target_col": target_col,
                "features_col": features_col.to_list(),
                "stratify": stratify,
                "random_seed": random_seed,
            },
            "data": {
                "train_data": train_data,
                "n_samples": n_samples,
            },
            "metrics": report[best_model],
        }
        with open(Path(save_dir) / "{}.json".format(best_model), "w") as f:
            f.write(json.dumps(metadata_report))

    elif not only_best:
        for key, value in models.items():
            metadata_report = {
                "run": {
                    "artifact_name": key + ".pkl",
                    "timestamp": datetime.now().strftime("%d/%m%Y, %H:%M%S"),
                },
                "model": {
                    "type": value.type.name,
                    "params": value.params,
                },
                "training": {
                    "target_col": target_col,
                    "features_col": features_col.to_list(),
                    "stratify": stratify,
                    "random_seed": random_seed,
                },
                "data": {
                    "train_data": train_data,
                    "n_samples": n_samples,
                },
                "metrics": report[key],
            }
            with open(Path(save_dir) / "{}.json".format(key), "w") as f:
                f.write(json.dumps(metadata_report))
