# PaDiM for Anomaly Detection.

This repository contains example codes of anomaly detection with PaDiM. Supported dataset is MVTec with the following format.

```
./data
├── test
│   ├── test_0.png
 ...
├── train
│   ├── good
│   │   ├── ok213.png
│    	...
└── val
    ├── good
    └── not-good
```

For convenience, you can use my utility scripts to create dataset automatically.
To use it put `archive` on this project root directory, and run the following command.

```bash
make create_dataset
```

This will split train data for validation for you.

NOTE: PaDiM only uses good cases to fit. So the train directory will only contain `good` directory. And all anomalies will be used for validation.
## How to run locally

### 1. Install dependency
Run the following command. This will install dependency automatically and initialize repository with pre-commit.

```bash
make init
```

### 2. Download your MVTec dataset

Download MVTec dataset and put it under `./data/`. Please make sure your dataset's format is compatible with this project.

### 3. Train and inference

Run the following command. This will call `src/main.py`.

```bash
make run
```

PaDiM use pre-trained model and just learn distribution from `good` images.
So the process will not take hours but will finished in 10 or 20 min.

## Run on Docker

Run the following command.

```bash
# Build image.
make build_docker
# Run container.
make run_docker
```


## Reference

https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier. PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization. https://arxiv.org/pdf/2011.08785

