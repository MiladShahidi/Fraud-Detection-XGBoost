# Summary

A full description of the project can be found [here](https://github.com/MiladShahidi/xgboost-fraud-detection/blob/master/XGBoost_Fraud_Detection.ipynb).

In this project I use the **Extreme Gradient Boosting (XGBoost)** algorithm to detect fradulent credit card transactions in a real-world (anonymized) dataset of european credit card transactions. I evaluate the performance of the model on a held-out test set and compare its performance to a few other popular classification algorithms, namely, **Logistic Regression, Random Forests and Extra Trees Classifier** (Geurts, Ernst, and Wehenkel 2006), and **show that a well-tuned XGBoost classifier outperforms all of them**.

The main challenge in fraud detection is the **extreme class imbalance** in the data which makes it difficult for many classification algorithms to effectively separate the two classes. **Only 0.172% of transactions are labeled as fradulent** in this dataset. I address the class imbalance by reweighting the data before training XGBoost (and by SMOTE oversamping in the case of Logistic regression).

Hyper-parameter tuning can considerably improve the performance of learning algorithms. XGBoost has many hyper-parameters which make it powerful and flexible, but also very difficult to tune due to the high-dimensional parameter space. Instead of the more traditional tuning methods (i.e. grid search and random search) that perform a brute force search through the parameter space, I use **Bayesian hyper-parameter optimization** (implemented in the hyperopt package) which has been shown to be more efficient than grid and random search (Bergstra, Yamins, and Cox 2013).

The full python code can be found [here](https://github.com/MiladShahidi/xgboost-fraud-detection/blob/master/XGBoost_Fraud_Detection.py).

Keywords: **XGBoost, Imbalanced/Cost-sensitive learning, Bayesian hyper-parameter tuning**