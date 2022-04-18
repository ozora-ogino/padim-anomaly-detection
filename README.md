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
│   └── not-good
│       ├── manipulated_front008.png
│    	...
└── val
    ├── good
    └── not-good
```


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

I referred the following project.

https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

