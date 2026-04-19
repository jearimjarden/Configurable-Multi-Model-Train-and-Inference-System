# 🚀 ML System with Robust Inference & Data Contract Enforcement

> A production-style machine learning system designed to ensure reliable training and inference through dynamic schema validation, artifact consistency checks, and failure-aware pipeline design.

## 🔥 Why This Project Stands Out

🧠 Dynamic schema generation from training metadata (no hardcoded contracts)
🔗 UUID-based artifact validation to prevent model–metadata mismatch
🔄 Dual inference workflows (batch processing & real-time service)
🛡️ Failure-aware pipeline (explicit validation, no silent errors)
⚙️ Fully config-driven architecture (no logic hardcoding)
🔄 Unified input abstraction (JSON, dict, DataFrame, CSV → one pipeline)
📊 End-to-end ML lifecycle (training → artifact → inference)

## ⚡ Quick Example

```python
from src.pipeline.inferencing import InferencePipeline

pipeline = InferencePipeline.from_config(config, logger, settings)

result = pipeline.predict({
    "Age": 30,
    "Income": 5000
})

print(result)
```

### Example Output

{
  "data_id": 1,
  "prediction": 1,
  "probability": 0.87
}

## 📌 What Problem This Solves

Prevents common ML failures:
- feature mismatch between training and inference
- inconsistent preprocessing pipelines
- incorrect artifact version usage
- silent schema inconsistencies

This system enforces a strict data contract + pipeline consistency + explicit validation to eliminate these failure modes.

## 🧠 Architecture

Training:
→ load config
→ load data
→ create preprocessing pipeline
→ train multiple models
→ evaluate models
→ select best model
→ save artifact (pipeline) + metadata

Inference:
→ load metadata
→ load artifact
→ generate validation schema dynamically
→ normalize input
→ validate input
→ align features
→ predict
→ generate report

## 🔄 Inference Modes

🟢 Batch Mode (CSV)
Input: CSV file
Output: Prediction report
Use case: bulk prediction / analytics

🔵 Service Mode (Real-Time)
Input: dict / JSON / DataFrame
Output: Prediction result
Use case: APIs / applications

⚙️ Configuration (.env)
PREDICT_SERVICE=true   # real-time mode
PREDICT_SERVICE=false  # batch mode

## 🔥 Feature List

### 🔄 Unified Input Handling

Supports multiple formats:

JSON string
Python dictionary
List of dictionaries
Pandas DataFrame
CSV file

All inputs are normalized into a unified internal format before validation and prediction.

### 🧠 Dynamic Schema Generation

Validation schema is generated directly from training metadata:

metadata → Pydantic model → runtime validation

Ensures:

strict consistency between training and inference
no duplicated schema definitions
automatic adaptation to new features

### 🔧 Data Normalization Layer

Before validation, input data is normalized:

type coercion (numeric ↔ categorical)
semantic alignment with training features
compatibility enforcement with model expectations

### 🆔 Data Traceability

Each input row is assigned a data_id:

input → validation → prediction → report

Enables:

traceable predictions
debugging at row level
structured reporting

### 📊 Prediction Output

Each prediction includes:

data_id → row identifier
prediction → model output
probability → confidence score

###🔗 Artifact–Metadata Validation

The system enforces strict consistency:

artifact.uuid == metadata.uuid

Mismatch → immediate failure.

Prevents:

incorrect model usage
version mismatch bugs
silent inference errors

### 🛡️ Failure-Aware Design

The system is designed to fail safely:

row with missing features are skipped or rejected (configurable)
all errors include structured context
no silent failures

### 📊 Structured Logging
JSON-based logs
stage-aware tracking (training, validation, inference)
contextual metadata (data_id, schema, errors)
optional file logging

## 📁 Structure

src/
  core/
  pipeline/
  services/
  data/
  io/
  tools/

## ⚙️ Configuration System

All behavior is controlled via config.yaml.

Key Sections
data → dataset paths
train → models, parameters, CV strategy, target
inference → artifact loading, threshold, feature handling
artifact → saving strategy

## 🚀 Run

Training:
python -m src.core.train

Inference:
python -m src.core.inference

## 🧩 Design Decisions
Only pre-defined models allowed → ensure reliability
Pipeline saved as artifact → guarantees preprocessing consistency
UUID validation → prevents artifact mismatch
Schema validation at inference → enforces strict data contract
Feature alignment:
extra features → dropped
missing features → optionally imputed
order → automatically aligned

## ⚠️ Limitations
Binary classification only
Single evaluation metric
No automated retraining pipeline


## 🗃️Detailed Configuration Sections

Configuration Sections
    1. Config.yaml
        a. data
            - train_path: path to the csv file used for training
            - inference_path: path to the csv file used for inference (prediction)
        b. train
            - model: dictionary of model's name and parameter
                -type: model's name, only prelisted model are accepted (look prelisted model section)
                -params: dictonary of parameter for selected model's type
            - stratify: enabled stratified cross-validation to maintain consistent class ratio accross fold, improving evaluation capability
            - n_cv: number of folds used in cross-validation to evaluate model performance
            - random_seed: control randomness to ensure reproducibility across cross-validation and model training
            - target_col: name of the column that will be used as label / target for model training
            - true_value: the positive value for target_col in case positive value are not automatically recognize for numpy
            - drop_features: a list of columns name that will be dropped before model's training
            - selections_metrics: name of selected metrics that will be produced and will be used to determined best model
            - missing_strategy: imputer's strategy for categorial data ("most_frequent" or "constant")
        c. inference
            - load_dir: directory path where artifact and metadata are stored
            - metadata_name: the name of metadata that will be used for model inference
            - allow_missing_features: allowing system to accept inference's data with missing features and impute it automatically
            - inference_report_path: directory path where prediction report will be saved
            - threshold: define threshold that determined predictions from probability
            - save_result: save or do not save inference report
        d. artifact
            - save_dir: directory path where artifact and metadata file are saved during training module
            - only_best: define wether training system only save best (according to selected metrics) or all trained model
    
    2. CLI
        a. logger / -l : select logging level
            - recognize input: debug, info, warning, error, critical
            - default: info

    3. ENV
        a. PREDICT_SERVICE: Enable inference service 
        b. SAVE_LOG: enable / disable file handler logger
        c. SAVE_LOG_LEVEL: logger level for file handler

    4. Pre-listed Model:
        a. LogisticRegression
        b. DecisionTreeClassifier
        c. RandomForestClassifier


Jearim Jarden