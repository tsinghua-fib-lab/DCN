# Dual Contrastive Network for Sequential Recommendation with User and Item-Centric Perspectives

This is our TensorFlow implementation for the paper:

Dual Contrastive Network for Sequential Recommendation with User and Item-Centric Perspectives

The code is tested under a Linux desktop with TensorFlow 1.12.3 and Python 3.6.8.



## Data Pre-processing



The script is `reco_utils/dataset/sequential_reviews.py` which can be excuted via:

```
python examples/00_quick_start/sequential.py --is_preprocess True
```

  

## Model Training

To train our model on `Amazon Toys` dataset (with default hyper-parameters): 

```
python examples/00_quick_start/sequential.py
```

## Misc

The implemention of self attention is modified based on *[TensorFlow framework of Microsoft](https://github.com/microsoft/recommenders)*.
