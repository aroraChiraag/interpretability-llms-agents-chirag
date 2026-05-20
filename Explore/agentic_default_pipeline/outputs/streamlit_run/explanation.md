## Headline result
The XGBoost model emerged as the top performer for identifying credit card defaulters, achieving a ROC-AUC of 0.779 and an F1-score of 0.530.

## Per-model metrics

**Random Forest**
The Random Forest model achieved an accuracy of 0.8089 and a balanced accuracy of 0.6697. It had a precision of 0.5970 and a recall of 0.4201, resulting in an F1-score of 0.4931. Its ROC-AUC was 0.7702, and average precision was 0.5486. Training took 5.111 seconds. The confusion matrix shows that the model correctly identified 4291 non-defaulters (True Negatives) and 557 defaulters (True Positives). However, it incorrectly flagged 376 non-defaulters as defaulters (False Positives) and missed 769 actual defaulters (False Negatives). For a bank, a higher precision means fewer non-defaulters are wrongly flagged, reducing unnecessary interventions, while a lower recall means more actual defaulters are missed, increasing potential losses.

**XGBoost**
The XGBoost model showed an accuracy of 0.7517 and a balanced accuracy of 0.7091. It achieved a precision of 0.4560 and a recall of 0.6327, leading to an F1-score of 0.5300. Its ROC-AUC was 0.7794, and average precision was 0.5658. Training took 45.122 seconds. From the confusion matrix, the model correctly identified 3666 non-defaulters (True Negatives) and 839 defaulters (True Positives). It made 1001 False Positive errors (incorrectly flagging non-defaulters) and 487 False Negative errors (missing actual defaulters). For a bank, this model's higher recall implies it catches more defaulters, potentially reducing financial losses, but its lower precision means more non-defaulters are wrongly flagged, which could lead to higher operational costs or customer dissatisfaction.

**Neural Network**
The Neural Network model achieved an accuracy of 0.8181 and a balanced accuracy of 0.6508. It had a precision of 0.6700 and a recall of 0.3507, resulting in an F1-score of 0.4604. Its ROC-AUC was 0.7670, and average precision was 0.5405. Training was the fastest at 4.437 seconds. The confusion matrix indicates that the model correctly identified 4438 non-defaulters (True Negatives) and 465 defaulters (True Positives). It had the fewest False Positives (229) but the most False Negatives (861). For a bank, this model's high precision means very few customers are wrongly inconvenienced, but its low recall means a significant number of actual defaulters are missed, posing a higher risk of financial loss.

## Feature importance

The models relied on different sets of features to make their predictions.
The **Random Forest** model found `UTIL_AVG` (average credit utilization), `UTIL_RECENT` (recent credit utilization), `PAY_0` (repayment status in September), `MEAN_DELAY` (average payment delay), and `BILL_TREND` (trend in bill amounts) as its top 5 most important features.
The **XGBoost** model identified `MAX_DELAY` (maximum payment delay) as overwhelmingly the most important feature, followed by `MEAN_DELAY`, `PAY_0`, `DELAY_TREND` (trend in payment delays), and `ZERO_PAYMENT_MONTHS` (number of months with zero payment) as its top 5.
The **Neural Network** model's feature importance, proxied by the sum of absolute weights from the first hidden layer, highlighted `PAY_2` (repayment status in August), `PAY_0`, `DELAY_TREND`, `PAY_AMT2` (amount paid in August), and `PAY_4` (repayment status in June) as its top 5.

Features that appeared in the top 5 for multiple models include:
*   `PAY_0`: Important for Random Forest, XGBoost, and Neural Network.
*   `MEAN_DELAY`: Important for Random Forest and XGBoost, and appeared in the top 10 for Neural Network.
*   `DELAY_TREND`: Important for XGBoost and Neural Network, and appeared in the top 10 for Random Forest.

It's clear that payment behavior (`PAY_0`, `PAY_2`, `PAY_4`, `DELAY_TREND`, `MAX_DELAY`, `MEAN_DELAY`, `ZERO_PAYMENT_MONTHS`) and credit utilization (`UTIL_AVG`, `UTIL_RECENT`) are consistently strong indicators across different model types.

## Reading the metrics

In the context of credit card default classification, we are trying to predict if a customer will default (Class 1) or not default (Class 0). The dataset has a class imbalance, with 23335 instances of "no default" (Class 0) and 6630 instances of "default" (Class 1). This means only about 22% of customers in our dataset are defaulters.

*   **Accuracy**: This is the proportion of total predictions that were correct. While it seems high for some models (e.g., Neural Network at 0.8181), it can be misleading in imbalanced datasets because a model could simply predict "no default" for most cases and still achieve high accuracy.
*   **Balanced Accuracy**: This metric accounts for class imbalance by averaging the recall obtained on each class. It gives a more reliable picture of a model's performance on both the majority and minority classes.
*   **Precision (for Class 1, Default)**: When the model predicts a customer will default, how often is it correct? High precision means fewer non-defaulters are wrongly flagged, which is good for avoiding unnecessary customer interventions or denying credit to good customers.
*   **Recall (for Class 1, Default)**: Of all the customers who actually defaulted, how many did the model correctly identify? High recall means the model catches most actual defaulters, which is crucial for minimizing financial losses for the bank.
*   **F1-score**: This is the harmonic mean of precision and recall. It provides a single score that balances both concerns, being particularly useful in imbalanced datasets where a trade-off between precision and recall often exists.
*   **ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)**: This measures the model's ability to distinguish between the two classes across all possible classification thresholds. A higher ROC-AUC (closer to 1) indicates better separability between defaulters and non-defaulters.
*   **Average Precision**: This is the area under the Precision-Recall curve. It is especially useful for imbalanced datasets and focuses on the model's performance at identifying the positive class (defaulters). A higher value indicates better performance.

Given the class imbalance (only 22% defaulters), metrics like Balanced Accuracy, F1-score, ROC-AUC, and Average Precision are more informative than raw Accuracy for evaluating how well a model identifies the minority class (defaulters).

## Caveats and next steps

*   **Class Imbalance Handling**: The dataset is imbalanced (22% defaulters). While some models (like Random Forest with `class_weight='balanced'`) attempt to address this, further techniques like SMOTE or adjusting decision thresholds could be explored to improve the detection of the minority class.
*   **Threshold Tuning**: The current metrics are based on a default classification threshold (usually 0.5). Adjusting this threshold could significantly alter the precision-recall trade-off, allowing the bank to prioritize minimizing false positives or false negatives based on business needs.
*   **Model Calibration**: It's important to assess if the predicted probabilities from the models accurately reflect the true likelihood of default. Uncalibrated probabilities can lead to poor decision-making, even with good ranking metrics like ROC-AUC.
*   **Fairness Audits**: Features like `SEX`, `EDUCATION`, and `MARRIAGE` are present in the dataset. It is crucial to conduct fairness audits to ensure the models do not inadvertently discriminate against specific demographic groups, even if these features are not explicitly used in training or are removed.

## Stage 4 tuning analysis

The XGBoost model underwent a tuning process.

-   **Technique applied**: The XGBoost model was tuned using **GridSearchCV**. This technique systematically explores a predefined set of hyperparameter combinations, evaluating each one using cross-validation to identify the optimal settings for a given scoring metric. In this case, the tuning aimed to maximize `recall`.
-   **Effect on metrics**: The GridSearchCV process resulted in a `best_cv_score` for recall of 0.6465. On the test set, the tuned XGBoost model achieved a recall of 0.6327, which is notably higher than the Random Forest (0.4201) and Neural Network (0.3507) models. This improvement in recall, however, came with a lower precision of 0.4560 compared to the other models. Despite this, its F1-score (0.5300) and ROC-AUC (0.7794) were the highest among all models.
-   **Plain-language explanation**: GridSearchCV essentially tried out many different configurations for the XGBoost model, such as the learning rate, maximum depth of individual trees, and number of trees. It then selected the configuration that was best at correctly identifying actual defaulters during cross-validation.
-   **Trade-offs introduced**: This tuning strategy prioritized maximizing recall, which means the model is now more effective at identifying customers who will actually default, thereby reducing potential financial losses for the bank. The trade-off is a lower precision, meaning more customers who would not have defaulted are incorrectly flagged. This could lead to increased operational costs from unnecessary interventions or potential customer dissatisfaction.
-   **Recommendation**: Given that the tuned XGBoost model achieved the highest F1-score and ROC-AUC, and significantly improved recall, it is the recommended model for deployment. This is particularly true if the business priority is to minimize financial losses by catching as many defaulters as possible. Further fine-tuning of the decision threshold could be considered to adjust the balance between precision and recall to align with specific business tolerance for false positives.