# Credit Card Fraud Prediction
Credit card fraud prediction comparing performance of random forest versus gradient boosting both implemented from scratch in python for educational purposes. 
sklearn was used for splitting as well as evaluation not for main model logic.
# Project Motivations
The main motivation for this project is to learn by doing and to see how the two models perform compared to each other.
I chose to use the models for credit card fraud prediction because it is a real problem in the world that models like these
can be used for. Since the dataset is highly imbalanced the goal is not to maximize accuracy because if the model predicts 
not fraud for every sample then it will get a high accuracy. Therefore, I wanted to maximize precision and recall to reduce false 
negatives and false positives.
# Dataset
Credit Card Fraud Detection on kaggle:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
Classes: 0=normal 1=fraud  
492 frauds out of 284,807 transactions, frauds account for 0.172% of all transactions
# Models
## Random Forest
The random forest implementation builds decision trees by minimizing gini impurity when creating splits. Bootstrap sampling 
is used to select what each tree gets to see, meaning each tree sees n random samples from a possible n samples with replacement, 
where n is the total number of samples in the training set. for each split feature subsampling is used, which means that 
each split considers a random subset of all possible features in my case the size of this subset is equal to the square 
root of the total number of features as an integer. The predictions of all trees for each sample is combined into one 
prediction using voting where whichever class is voted for by more trees wins the vote. A threshold is used to compute
the class prediction by taking the number of fraud votes over the number of trees and comparing it to the threshold.
## Gradient Boosting
The gradient boosting model starts with the log-odds of the fraud rate (mean of y) as the initial prediction for the model.
The model then adds trees sequentially that will use the training data to try and reduce the difference between the 
predictions and the actual, this difference is the residuals. A numerically stable version of the sigmoid function is used
to convert the log-odds into probabilities. the hyperparameters learning-rate, number of estimators, max depth, and 
minimum samples is used to reduce over-fitting. Since the model is built to return the raw probabilities given the input data 
threshold tuning can be used to find the best threshold for classification.
# Training and Testing Approach
sklearn is used to split the data into training and testing subsets while keeping the original fraud ratio. For the 
training set the number of normal transactions is reduced to create more balanced training data.
# Evaluation
sklearn is used to compute the classification report. Because the dataset is highly imbalanced, the fraud-class precision, 
recall, and F1-score are more meaningful than accuracy alone.
## Random Forest

| Metric | Fraud Class (1) |
|---|---|
| Precision | 0.8077 |
| Recall | 0.8571 |
| F1-Score | 0.8317 |

Overall Accuracy: `0.9994`

<details>
<summary>Full Classification Report</summary>

```text
              precision    recall  f1-score   support

           0     0.9998    0.9996    0.9997     56864
           1     0.8077    0.8571    0.8317        98

    accuracy                         0.9994     56962
   macro avg     0.9037    0.9284    0.9157     56962
weighted avg     0.9994    0.9994    0.9994     56962
```

</details>

---

## Gradient Boosting

| Metric | Fraud Class (1) |
|---|---|
| Precision | 0.8039 |
| Recall | 0.8367 |
| F1-Score | 0.8200 |

Overall Accuracy: `0.9994`

<details>
<summary>Full Classification Report</summary>

```text
              precision    recall  f1-score   support

           0     0.9997    0.9996    0.9997     56864
           1     0.8039    0.8367    0.8200        98

    accuracy                         0.9994     56962
   macro avg     0.9018    0.9182    0.9098     56962
weighted avg     0.9994    0.9994    0.9994     56962
```

</details>

Both models achieved a very high accuracy of 99.94% which is just because of the class imbalance and not reflective of 
the performance of the model. The recall scores for fraud on both shows that the models are catching more than 83% of the 
fraud cases. The precision scores for fraud also shows that when the models predict fraud around 80-81% of them are actually fraud. Random
forest managed to do a little better on recall and F1, which means it was able to catch the fraud 2-3% more than gradient 
boosting while maintaining a similar precision, ie no more false positives than gradient boosting.

For Threshold tuning gradient boosting achieved a best threshold of 0.53 with a F1 score of 0.82.

# How to run
