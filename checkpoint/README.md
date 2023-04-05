## README for data

### Pre-trained model 

You can download the trained model from this [OneDrive link](https://1drv.ms/u/s!Ah9r82oejjV8n0tf5MY1TC8bfGfA?e=pxOG6Y) and unzip the archive to this folder. The archive contains the following files: 

1. Model checkpoint files. These three files are the best three models on CASF-2016 from the experiments.
  - `model1.pt`
  - `model2.pt` (used in paper)
  - `model3.pt` 

2. Prediction results on CASF-2016 and CASF-2013.
  - `model1_CASF2016.csv`
  - `model1_CASF2013.csv`
  - `model2_CASF2016.csv` (used in paper)
  - `model2_CASF2013.csv`
  - `model3_CASF2016.csv` 
  - `model4_CASF2013.csv`


### Training data

The preprocessed data for training is downloaded on the fly. Check bash scripts in `../Dynaformer/examples/` for details. The zipped processed data will be downloaded to the path specified by `--data-path` parameter. By default, this parameter will be `../data` directory.

