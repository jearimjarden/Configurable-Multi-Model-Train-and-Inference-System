# Configurable Multi Model Training and Inference System

## Description
A configurable machine learning system that separates training and inference workflows. 
It supports multi-model training, automated model selection based on evaluation metrics, 
and a structured pipeline for consistent preprocessing, training, and prediction.

## Features List
    1. Separated train and inference module
    2. Configurable train and inference modul via config.yaml
    3. JSON structured logger and file handler
    4. Multi-model train compability
    5. Automatically produced the best model with selected metrics options
    6. Custom true value for selected label 
    7. Custom metrics for train evaluation
    8. Custom threshold for prediction result in inference module
    9. Inference service that accept json or panda's dataframe

## Configuration Sections
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
        a. PREDICT_SERVICE: Enable inference service panda's dataframe or json input / using selected csv file as inference input
        b. SAVE_LOG: enable / disable file handler logger
        c. SAVE_LOG_LEVEL: logger level for file handler

    4. Pre-listed Model:
        a. LogisticRegression
        b. DecisionTreeClassifier
        c. RandomForestClassifier


## How it Works
    1. Training Module:
        a. Flowchart:
            load config -> load training data -> create training preprocessor -> evaluate data using selected model -> fit model -> save artifact and metadata
        b. Input:
            - Training Data: csv data used for training
        c. Output:
            - Artifact : store fitted model
            - Metadata : store information for fitted model

    2. Inference Module:
        - Flowchart:
            load config -> load metadata -> load artifact -> load inference data -> validate and align data using metadata information -> predict inference data using loaded artifact -> save prediction report
        - Input:
            - Artifact: stored fitted model
            - Metadata: stored information for fitted model
            - Inference Data: csv data that will be predicted
        - Output:
            - Prediction Report: store prediction information, prediction and probability of each predicted data


## Error Handling
    Error will be logged including message and stage

    1. All exception that catch in pipeline will be logged (training_pipeline.py and inference_pipeline.py)
    2. Unexpected will catch by entry points (train.py and inference.py)
    3. Exit Code
        - 0: Module exited with no error
        - 1: Expected error occured 
        - 2: Unexpected error occured
    4. Error Name:
        - ConfigError: Invalid config structure or value
        - DataError: error caused by the inputted data for training or inference
        - TrainingError: error caused in training pipeline
        - InferenceError: error caused in inference pipeline
        - MetadataError: error caused while loading metadata
        - ArtifactError: error caused while loading artifact
        - LoggedError: a flag for logged error to avoid multiple logging

## How to Run
    1. Training Module:
    bash '
    pip install -r requirements.txt
    python3 -m src.core.train'

    2. Inference Module:
    bash '
    pip install -r requirements.txt
    pyhthon3 -m src.core.inference'

    Notes:
    - Read Configuration Section before running the module
    - training and inference module are controlled via config.yaml
    - CLI are used to control logging level ('-l' or '-logger')


## Project Structure
    - src
        - core
            - train.py
            - inference.py
        - pipeline
            - training.py
            - inferencing.py
        - services
            - data_loader.py
            - models.py
            - preproecessor.py
            - IO.py
        - tools
            - exceptions.py
            - cli.py
            - loader.py
            - schemas.py
    - data
        - artifacts
        - reports
        - test
        - train
    - logs
    - config.yaml
    - README.md


## Design decisions
    1. Only pre-listed model are accepted to ensure model reability
    2. All input from outside system are validated using pydantic ensuring same field with training's field
    3. Artifacts stored pipeline that contain preprocessor and model, to ensure that artifact always stored a complete artifact file
    4. UUID are introduced for artifact and metadata, and will be used for inference module to check artifact fitness to metadata
    5. Folder are separated: core (entry points), pipeline (execution flow and services orchestrator for each module), services (system's logic), tools (tools that support system)
    6. Extra and missing features are produce by comparing features list in metadata and inference data features list
    7. Extra features always dropped 
    8. Metrics selection and parameters are only checked for the data type and incoming error caused by them are handled in system's service
    9. Inference features order are automatically re-ordered to ensure model prediction realibility
    10. Inference module has 2 options: 1. service to accept panda's dataframe or json,     2. data input from csv file
    11. All input data for inference module are converted to dict then checked by pydantic so extra columns will be ignore
    12. Normalization happened for input data from pandas dataframe to ensure correctness of field's type
    

## Limitations
1. Only work for classification model with binary label / target
2. Artifact validation only check wether model pipeline is exist in artifact or not
3. Only accept single selections metrics could not accept a list of selections metrics
