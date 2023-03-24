# Time-Series Anomaly Detection 
Apply Time-Series Anomaly Detection to the data with different models for 2022 winter individual research.

## Development Environment
* OS: Linux
* Python: 3.9
* Tensorflow: 2.5.2
* GPU: RTX 3090

## Dataset
I used [SMAP](https://smap.jpl.nasa.gov/data/), [SMD](https://github.com/NetManAIOps/OmniAnomaly), [UCR](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) datasets for this study. 
I preprocessed the data for training, and you can find this data [here](https://drive.google.com/drive/folders/1k08NWe6zHolSHR6a5LzLFMHzR6T7BKlb?usp=sharing).

To reproduce the results, make "datasets" directory at the root, and put these 3 folders into the "datasets" folder.

You can find more information in [dataloader.py](https://github.com/chlehdwon/Timeseries_Anomaly_Detection/blob/main/dataloader.py)

## Models
* CNN-based AE
* LSTM-based AE
* LSTM-based VAE
* LSTM-based GAN
* GRU-based AE
* GRU-based VAE
* GRU-based GAN

I implemented these 7 models(simply) by using keras and apply these models to each dataset.

You can find more information in each notebook files

## Evaluations
I applied many methods of computing thresholds and evaluating them. 
For evaluation, I tried these 2 methods.

* Method1: If any/all included time steps have the anomalies data => abnormal

* Method2: If more than half of included time steps have anomaly data => abnormal

You can find more information in [evaluator.py](https://github.com/chlehdwon/Timeseries_Anomaly_Detection/blob/main/evaluator.py)

## Results
<img src="https://user-images.githubusercontent.com/68576681/227614971-d2f3cee1-22d7-4bb2-ae30-fd2cfcddeb35.png" width="100%" height="360">

