import pickle
import os
from pathlib import Path
from datetime import datetime
import json
from sklearn.pipeline import Pipeline


def create_artifact(
    best_model: str,
    fitted_model: dict[str, Pipeline],
    only_best: bool,
    save_dir: str,
) -> None:
    path = Path(save_dir)
    os.makedirs(path, exist_ok=True)
    if only_best:
        with open(path / "best_{}.pkl".format(best_model), "wb") as f:
            pickle.dump(fitted_model[best_model], f)

    elif not only_best:
        for name, pipeline in fitted_model.items():

            if name == best_model:
                save_name = "best_{}.pkl".format(name)
            else:
                save_name = "{}.pkl".format(name)

            with open(path / save_name, "wb") as f:
                pickle.dump(pipeline, f)


def create_metadata(
    best_model,
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
) -> None:
    if only_best:

        metadata_report = {
            "run": {
                "artifact_name": best_model + ".pkl",
                "timestamp": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
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
        with open(Path(save_dir) / "best_{}.json".format(best_model), "w") as f:
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

            if key == best_model:
                save_name = "best_{}.json".format(key)
            else:
                save_name = "{}.json".format(key)

            with open(Path(save_dir) / save_name, "w") as f:
                f.write(json.dumps(metadata_report))
