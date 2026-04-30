## Headline result

The Random Forest model is the best performing model in this pipeline, achieving a ROC-AUC of 0.7658704439926587 and an F1-score of 0.4951627088830255.

## Per-model metrics

**Random Forest**
The Random Forest model achieved an accuracy of 0.8086666666666666 and a balanced accuracy of 0.671045533908578. For predicting default, it had a precision of 0.5945089757127772, a recall of 0.4242652599849284, and an F1-score of 0.4951627088830255. Its ROC-AUC was 0.7658704439926587 and average precision was 0.5441871803260725. The model took 4.411 seconds to train.

Looking at the confusion matrix, the model correctly identified 4289 non-defaulters (true negatives) and 563 defaulters (true positives). However, it incorrectly flagged 384 non-defaulters as defaulters (false positives) and missed 764 actual defaulters, classifying them as non-defaulters (false negatives). For a bank, the trade-off between precision and recall implies that while the model is reasonably accurate when it predicts a default, it still misses a significant portion of actual defaulters, which could lead to financial losses.

## Feature importance

The Random Forest model relies most heavily on `PAY_0` (payment status in September) with an importance of 0.10959595205778246. This is followed by `BILL_AMT1` (bill amount in September) at 0.06096650774387379, `LIMIT_BAL` (credit limit) at 0.058700657671416204, `PAY_AMT1` (amount paid in September) at 0.0543241775084194, and `AGE` at 0.05313781094186449. These features indicate that recent payment behavior, current bill amounts, credit limit, and age are the most influential factors for this model in predicting credit default.

## Reading the metrics

In the context of credit card default prediction, where '1' means default and '0' means no default:

*   **Precision** tells us, out of all the customers the model *predicted* would default, what percentage actually did default. A high precision means the bank won't incorrectly flag too many reliable customers, which is good for customer relations.
*   **Recall** tells us, out of all the customers who *actually* defaulted, what percentage the model correctly identified. A high recall means the bank is catching most of the actual defaulters, which is crucial for minimizing financial risk.
*   **F1-score** is a balance between precision and recall. It's especially useful in situations like this, where the number of defaulters (6636) is much smaller than non-defaulters (23364) in the dataset (an imbalanced class distribution). A good F1-score indicates the model performs well on both aspects.
*   **ROC-AUC** (Receiver Operating Characteristic - Area Under the Curve) measures the model's ability to distinguish between defaulters and non-defaulters across all possible classification thresholds. A higher value (closer to 1) means better discrimination.
*   **Average Precision** summarizes the precision-recall curve. Like F1-score, it's particularly informative for imbalanced datasets because it focuses on the model's performance on the positive class (default).

The class balance in our dataset shows that only 6636 out of 30000 customers (approximately 22.12%) defaulted. This imbalance means that a model could achieve high overall accuracy by simply predicting "no default" most of the time. Therefore, metrics like F1-score, ROC-AUC, and Average Precision are more reliable indicators of true performance than simple accuracy in this scenario.

## Caveats and next steps

*   **Class Imbalance Handling:** While the Random Forest model used `class_weight="balanced"`, the dataset remains imbalanced. Further exploration of techniques like SMOTE (Synthetic Minority Over-sampling Technique) or other advanced sampling methods could potentially improve the model's ability to identify defaulters.
*   **Threshold Tuning:** The current metrics are based on a default classification threshold. Depending on the bank's risk appetite – whether it prioritizes minimizing false alarms (higher precision) or catching as many defaulters as possible (higher recall) – the classification threshold can be adjusted to better align with business objectives.
*   **Fairness Audits:** Features such as `SEX`, `EDUCATION`, and `MARRIAGE` are included in the dataset. It is crucial to conduct fairness audits to ensure the model does not exhibit biased predictions or disproportionately impact specific demographic groups, which could lead to ethical and regulatory concerns.
*   **Model Calibration:** For real-world applications like setting credit limits or interest rates, it's important that the model's predicted probabilities are well-calibrated, meaning a predicted probability of 0.7 for default truly corresponds to a 70% chance of default. Calibration checks and adjustments should be performed.