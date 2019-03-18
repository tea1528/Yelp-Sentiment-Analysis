# Sentiment Analysis on Yelp Dataset
This project contains the code for COMP4332 Project 1 and COMP4901K Project 2 which were on sentiment analysis on multi-label reviews (predicting stars from 1 to 5).

The data for this project is a segment of [Yelp Dataset](https://www.yelp.com/dataset) by only using 100,000 for training set and 10,000 for validation and test set respectively. The data split is illustrated in the jupyter notebook in `data` folder. 

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

## Reference
This data contains code from https://github.com/prakashpandey9/Text-Classification-Pytorch.
