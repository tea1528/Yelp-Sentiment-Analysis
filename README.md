# Sentiment Analysis on Yelp Dataset
This project contains the code for COMP4332 Project 1. 

The data for this project is a segment of [Yelp Dataset](https://www.yelp.com/dataset) by only using 100,000 for training set and 10,000 for validation and test set respectively.

You can start training by running `src/main.py` and run inference using `src/test.py` which will store a prediction on the test set. 

## Folder Structure

```
data\
  Yelp_split.ipynb
results\
logs\
src\
  models\
    LSTM.py
    RCNN.py
    selfAttention.py
    LayerNorm.py
  main.py
  test.py
  load_data.py
  cls.py
config.yaml
requirements.txt
```
