# Hybrid inference: Inferring fast and slow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
![status](https://img.shields.io/badge/status-development-orange)


## Getting started
To install the relevant packages:
```bash
pip install -r requirements.txt
```

## Running
``` bash
python -m scripts.hybrid
```

## Dataset
[There is an issue with the torch MNIST loader](https://stackoverflow.com/questions/66577151/http-error-when-trying-to-download-mnist-data). You can download the dataset manually:
```bash
bash dl_mnist.sh
```