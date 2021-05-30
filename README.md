Master's Thesis
===============
## Estimation of arrival times for shipping

A machine learning model trains the estimation of arrival times (ETA) in shipping on data from the Danish Automatic Identification (AIS) System

### Pipeline:
1. Data preprocessing
1.1. Identification of drives per port and MMSI
1.2. Handling NaN, undefined, One-Hot-Encoding
1.3. Automatic label generation
2. Model training from scratch for ports (base-models)
3. Transfer of models to each other port
4. Evaluation (MAE, plots)
