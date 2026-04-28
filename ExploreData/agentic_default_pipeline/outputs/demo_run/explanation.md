## Headline result

The best performing model in this evaluation is **xgboost**, achieving a ROC-AUC of 0.7804886446228402 and an F1 score of 0.4694973157637872.

## Per-model metrics

### Random Forest
The Random Forest model achieved an accuracy of 0.8086666666666666, a balanced accuracy of 0.671045533908578, precision of 0.5945089757127772, recall of 0.4242652599849284, an F1 score of 0.4951627088830255, a ROC-AUC of 0.7658704439926587, and an average precision of 0.5441871803260725. It completed training in 4.401 seconds. Its confusion matrix shows 4289 True Negatives (correctly identified non-defaulters), 384 False Positives (non-defaulters wrongly flagged as defaulters), 764 False Negatives (defaulters wrongly flagged as non-defaulters), and 563 True Positives (correctly identified defaulters). With a higher recall than precision, this model is better at identifying a larger proportion of actual defaulters, even if some of its positive predictions are incorrect. For a bank, this implies a focus on minimizing missed defaulters, potentially at the cost of flagging some good customers.

### XGBoost (using sklearn GradientBoostingClassifier)
The XGBoost model, implemented using `sklearn.GradientBoostingClassifier` as a substitute, achieved an accuracy of 0.8188333333333333, a balanced accuracy of 0.6554494376858448, precision of 0.6662049861495845, recall of 0.3624717407686511, an F1 score of 0.4694973157637872, a ROC-AUC of 0.7804886446228402, and an average precision of 0.5551717828515215. It trained in 37.408 seconds. The confusion matrix for XGBoost shows 4432 True Negatives, 241 False Positives, 846 False Negatives, and 481 True Positives. This model prioritizes precision over recall, meaning that when it predicts a default, it is more often correct, but it misses a larger proportion of actual defaulters. A bank using this model would be more concerned with avoiding false alarms (wrongly flagging good customers) than with catching every single defaulter.

### Neural Network
The Neural Network model achieved an accuracy of 0.817, a balanced accuracy of 0.6540026714740083, precision of 0.6566347469220246, recall of 0.3617181612660136, an F1 score of 0.46647230320699706, a ROC-AUC of 0.7663087553746765, and an average precision of 0.5399980395508097. It trained in 6.409 seconds. Its confusion matrix indicates 4422 True Negatives, 251 False Positives, 847 False Negatives, and 480 True Positives. Similar to XGBoost, the Neural Network also shows a higher precision than recall. This suggests it is also more conservative in flagging potential defaulters, aiming to reduce the number of good customers who are incorrectly identified as high-risk.

## Feature importance

*   **Random Forest**'s top 5 features are: `PAY_0` (0.10959595205778246), `BILL_AMT1` (0.06096650774387379), `LIMIT_BAL` (0.058700657671416204), `PAY_AMT1` (0.0543241775084194), and `AGE` (0.05313781094186449).
*   **XGBoost** (GradientBoostingClassifier) identifies `PAY_0` (0.6152733860132175) as overwhelmingly the most important feature, followed by `PAY_2` (0.07188823264837557), `BILL_AMT1` (0.03979318645954638), `PAY_3` (0.033181385991350607), and `LIMIT_BAL` (0.029859051488858587).
*   For the **Neural Network**, feature importance is proxied via the sum of absolute weights from the first layer (`|W1|` sum across hidden units). Its top 5 features are: `PAY_0` (12.581882062463814), `SEX` (10.973870937115752), `PAY_AMT2` (10.736538940548252), `LIMIT_BAL` (10.620389638214235), and `PAY_2` (10.200050942828279).

Across all models, `PAY_0` (repayment status in September) consistently ranks as the most important feature, highlighting its critical role in predicting default. `LIMIT_BAL` (credit limit) appears in the top 5 for all three models, and `BILL_AMT1` (bill amount in September) is important for Random Forest and XGBoost. `PAY_2` (repayment status in August) is also a significant feature for XGBoost and the Neural Network.

## Reading the metrics

In the context of credit card default prediction, where '1' means default and '0' means no-default:
*   **Precision** measures how many of the customers predicted to default actually did default. A high precision means fewer "false alarms" – fewer good customers are wrongly flagged as high-risk.
*   **Recall** (also known as sensitivity) measures how many of the actual defaulters were correctly identified by the model. A high recall means the model catches most of the people who will default, minimizing missed defaulters.
*   **F1-score** is the harmonic mean of precision and recall, providing a single metric that balances both. It's particularly useful when dealing with imbalanced datasets, as a model might achieve high accuracy by simply predicting the majority class, but F1 would reveal its poor performance on the minority class.
*   **ROC-AUC** (Receiver Operating Characteristic - Area Under the Curve) measures the model's ability to distinguish between the two classes across all possible classification thresholds. A higher ROC-AUC indicates better overall discriminative power.
*   **Average Precision** summarizes the precision-recall curve, providing a single value that is more sensitive to the performance on the minority class than ROC-AUC, especially with imbalanced data.

The Data Agent reported a class balance of approximately 78% non-default (0) and 22% default (1). This moderate imbalance means that a model could achieve a seemingly high accuracy (e.g., 78%) by simply predicting "no default" for everyone. Therefore, metrics like F1, ROC-AUC, and Average Precision are more reliable indicators of true performance in identifying the minority class (defaulters).

## Caveats and next steps

*   **Class Imbalance Handling:** The dataset exhibits a moderate class imbalance (78% non-default, 22% default). While some models might inherently handle this better, further investigation into techniques like oversampling (e.g., SMOTE), undersampling, or using cost-sensitive learning could improve the detection of defaulters (recall) without excessively sacrificing precision.
*   **Threshold Tuning:** The current metrics are based on a default classification threshold (usually 0.5). Depending on the bank's specific business objectives (e.g., minimizing financial loss from defaulters vs. retaining good customers), the classification threshold could be adjusted to favor higher precision or higher recall.
*   **Model Calibration:** It's crucial to assess if the predicted probabilities from the models are well-calibrated, meaning that a predicted probability of, say, 0.7 truly corresponds to a 70% chance of default. This is important for downstream decision-making and risk assessment.
*   **Fairness Audits:** Given that features like `SEX`, `EDUCATION`, and `MARRIAGE` are present and `SEX` appeared as a top feature for the Neural Network, it is essential to conduct fairness audits. This involves evaluating model performance (e.g., false positive rates, false negative rates) across different demographic groups to ensure the model does not exhibit biased outcomes or perpetuate historical inequalities.