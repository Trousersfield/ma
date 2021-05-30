Master's Thesis
===============
## Estimation of arrival times for shipping

A machine learning model trains the estimation of arrival times (ETA) in shipping on data from the Danish Automatic Identification (AIS) System

### Pipeline:
1. Data preprocessing
    1.1. Identification of drives per port and MMSI
    1.2. Handling NaN, undefined, One-Hot-Encoding
    1.3. Automatic label generation
2. Model training from scratch for ports
    2.1. Architecture [InceptionTime](https://arxiv.org/abs/1909.04939) + Dense
    2.2. One model for each port (base-model)
3. Transfer of models to each other port
    3.1. Re-Train certain layers
    3.2. Iterate with different layers and less data
5. Evaluation (MAE, plots)
