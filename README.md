# For competition  

## Dataset  
[data_EEG_AI.mat](https://www.dropbox.com/s/2ug002c1btxkvvg/data_EEG_AI.mat?dl=0)  
```bash
wget https://www.dropbox.com/s/2ug002c1btxkvvg/data_EEG_AI.mat?dl=1
```
put file in `./data_EEG_AI.mat`

## Environment
```bash
conda env create -f environment.yml
conda activate egg
```

## To evaluate model
download pretrain [here](https://drive.google.com/file/d/1w6CuNZurj6nHmdLc-cwS3rGG9K28INmq/view?usp=sharing)
```bash
python info.py
```

## To see info of dataset and environment
```bash
python info.py
```

## To train
```bash
python train.py
```
