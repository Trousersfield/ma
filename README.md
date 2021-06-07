Master's Thesis
===============
## Estimation of arrival times for shipping

A machine learning model trains the estimation of arrival times (ETA) in shipping on data from the Danish Automatic Identification (AIS) System

### Pipeline:
1. Data preprocessing
    1. Identification of drives per port and MMSI
    2. Handling NaN, undefined, One-Hot-Encoding
    3. Automatic label generation
2. Model training from scratch for ports
    1. Architecture <a href="https://arxiv.org/abs/1909.04939" target="_blank">InceptionTime</a> + Dense
    2. One model for each port (base-model)
3. Transfer of models to each other port
    1. Re-Train certain layers
    2. Iterate with different layers and less data
4. Evaluation (MAE, plots)

### Preview:
#### MAE of Ports
- Base-Traning on diagonal
- Transfer from Port X to Port Y

