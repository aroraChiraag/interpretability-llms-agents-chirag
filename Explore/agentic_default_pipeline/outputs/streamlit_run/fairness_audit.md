## Executive Fairness Summary

All three models (Neural Network, Random Forest, and XGBoost) demonstrate critical "Bias Red Flags" with Disparate Impact Ratios below 0.80 for at least two protected attributes. The Neural Network exhibits the most severe bias, with a Disparate Impact Ratio of 0.43 against 'Undocumented' and 'Other' marriage statuses.

## Model Comparison: Outcome Disparities

### Disparate Impact
The Disparate Impact metric measures the ratio of the selection rate for a protected group to the selection rate of a reference group. A ratio below 0.80 indicates a potential disparate impact.

*   **SEX**: All models pass the 80% rule for 'SEX'. The Neural Network shows the best Disparate Impact for 'SEX' at 0.8509 (Female selection rate 0.1084, Male selection rate 0.1274), indicating the smallest disparity between Female and Male applicants. Random Forest is at 0.8343, and XGBoost is at 0.8244.
*   **AGE**: All models fail the 80% rule for 'AGE'. XGBoost exhibits the least severe disparity with a Disparate Impact of 0.765 (relative to the '31-45' age group, which has the lowest selection rate of 0.277). The Random Forest model shows the worst Disparate Impact for 'AGE' at 0.6252.
*   **MARRIAGE**: All models fail the 80% rule for 'MARRIAGE'. XGBoost again shows the least severe disparity with a Disparate Impact of 0.7742 (relative to 'Other' marriage status, which has the lowest selection rate of 0.2632). The Neural Network demonstrates the most significant disparity for 'MARRIAGE' at 0.43, indicating a severe bias against 'Undocumented' and 'Other' marriage statuses.

### Equalized Odds
Equalized Odds assesses if a model's False Positive Rate (FPR) and True Positive Rate (TPR) are consistent across different demographic groups. A smaller gap indicates better fairness.

*   **False Positive Rate (FPR) Consistency**:
    *   For 'SEX', the Neural Network has the smallest FPR gap at 0.0132, indicating it is most consistent in wrongly flagging safe customers across Female and Male groups. XGBoost has the largest FPR gap at 0.0609.
    *   For 'AGE', the Random Forest has the smallest FPR gap at 0.0232, showing better consistency across age groups. XGBoost has the largest FPR gap at 0.0876.
    *   For 'MARRIAGE', the Random Forest has the smallest FPR gap at 0.0645. The Neural Network has the largest FPR gap at 0.2379, indicating significant inconsistency in wrongly flagging safe customers across marriage statuses.
*   **True Positive Rate (TPR) Consistency**:
    *   For 'SEX', the Neural Network has the smallest TPR gap at 0.0109.
    *   For 'AGE', the Random Forest has the smallest TPR gap at 0.0928.
    *   For 'MARRIAGE', the Neural Network has a very large TPR gap of 0.3665, indicating significant inconsistency in correctly identifying defaulting customers across marriage statuses.

### Predictive Equality
Predictive Equality focuses on consistent False Positive Rates (FPR) across groups.
The Neural Network, despite its complexity, does not consistently lead to lower bias in terms of predictive equality. While it has the best FPR gap for 'SEX' (0.0132), it exhibits the worst FPR gap for 'MARRIAGE' (0.2379). The simpler Random Forest model demonstrates better predictive equality for 'AGE' (FPR gap 0.0232) and 'MARRIAGE' (FPR gap 0.0645) compared to both the Neural Network and XGBoost.

## Architecture-Specific Fairness Observations

*   **Neural Network**: The "Black Box" nature of the Neural Network makes it challenging to explain the observed disparities, particularly the severe Disparate Impact of 0.43 for 'MARRIAGE' status. The feature importance being proxied via `|W1| sum across hidden units` provides some insight into feature influence but does not directly explain *how* the model arrived at such disparate outcomes for specific groups.
*   **Random Forest**: The implementation of `class_weight='balanced'` aimed to improve the model's ability to correctly identify the minority class (defaults). While this technique can improve recall, its impact on fairness is mixed. For the Random Forest, it resulted in a lower overall recall (0.4201) compared to XGBoost, but it achieved the best FPR gaps for 'AGE' (0.0232) and 'MARRIAGE' (0.0645), suggesting some improvement in predictive equality for these attributes. However, it still failed the 80% rule for Disparate Impact on 'AGE' and 'MARRIAGE'.
*   **XGBoost**: XGBoost's aggressive optimization, specifically using `GridSearchCV` with `scoring=recall`, prioritized the model's ability to catch defaults. This optimization likely contributed to its highest recall (0.6327) among the models. However, this focus on recall appears to have exacerbated bias against minority groups in terms of predictive equality, as evidenced by its worst FPR gaps for 'SEX' (0.0609) and 'AGE' (0.0876). While its Disparate Impact scores for 'AGE' and 'MARRIAGE' are numerically closer to the 0.80 threshold compared to the other models, they still represent significant fairness concerns.

## The Fairness-Accuracy Trade-off

*   **XGBoost**: While XGBoost has the highest Recall (63.27%), it also shows Disparate Impact ratios below 0.80 for 'AGE' (0.765) and 'MARRIAGE' (0.7742). Its False Positive Rate gaps are also the highest for 'SEX' (0.0609) and 'AGE' (0.0876), indicating a trade-off where higher predictive power comes with increased outcome disparities for certain groups.
*   **Random Forest**: The Random Forest model achieves a Recall of 42.01%. It passes the 80% rule for 'SEX' (0.8343) but fails for 'AGE' (0.6252) and 'MARRIAGE' (0.6477). Notably, it demonstrates the best FPR gaps for 'AGE' (0.0232) and 'MARRIAGE' (0.0645), suggesting a better balance in avoiding false positives across these groups, albeit with lower overall recall.
*   **Neural Network**: The Neural Network has the lowest Recall (35.07%). While it shows the best Disparate Impact for 'SEX' (0.8509) and the best FPR gap for 'SEX' (0.0132), it exhibits a critically low Disparate Impact for 'MARRIAGE' (0.43) and the worst FPR gap for 'MARRIAGE' (0.2379). This indicates that its lower predictive power does not consistently translate to better fairness outcomes across all protected attributes.

## Final Audit Recommendation

Based on a balance of compliance, practicality, and explainability, the **XGBoost model** is recommended for deployment, with a strong caveat for immediate and significant fairness mitigation.

1.  **Compliance**: All models fail the 80% fairness rule for 'AGE' and 'MARRIAGE'. However, XGBoost's Disparate Impact scores for these attributes (0.765 for 'AGE' and 0.7742 for 'MARRIAGE') are numerically closer to the 0.80 threshold compared to the Random Forest and Neural Network models, indicating less severe violations in terms of selection rates.
2.  **Practicality (Recall)**: XGBoost demonstrates the highest recall (63.27%), which is crucial for the bank's ability to identify defaulting customers and manage credit risk effectively. The other models have substantially lower recall rates.
3.  **Explainability**: While not as inherently interpretable as a simple decision tree, XGBoost models are generally more interpretable than Neural Networks, especially when using techniques like SHAP or LIME for post-hoc explanations. The Neural Network's "Black Box" nature makes it harder to justify denials, particularly given its severe bias for 'MARRIAGE'.

It is explicitly noted that the XGBoost model was tuned using `GridSearchCV` with `scoring=recall`, which optimized for predictive performance. The reported fairness scores are the result of this tuning. While this tuning likely improved recall, it may have worsened fairness metrics, as indicated by its higher FPR gaps for 'SEX' and 'AGE' compared to other models. No baseline fairness scores for an untuned XGBoost model are available for direct comparison.

## Mitigation Next Steps

To address the identified biases in the chosen XGBoost model, the following mitigation steps are recommended:

1.  **Post-Processing: Optimized Decision Thresholds**: Implement group-specific decision thresholds. Instead of a single global threshold (e.g., 0.5), analyze the score distributions for different groups within 'AGE' and 'MARRIAGE' and adjust thresholds to achieve more equitable False Positive Rates or True Positive Rates across these groups. This can directly improve Equalized Odds.
2.  **In-Processing: Fair-Constrained Optimization**: Explore re-training the XGBoost model with fairness constraints integrated into the optimization process. Techniques like "Adversarial Debiasing" or "Regularization with Fairness Constraints" can be used to penalize the model for generating disparate outcomes during training, aiming to improve fairness metrics (e.g., Disparate Impact or Equalized Odds) while maintaining predictive performance.
3.  **Data Augmentation/Re-sampling (Targeted)**: For the 'MARRIAGE' attribute, where the Neural Network showed a critical Disparate Impact, and where XGBoost still has a failing DI, consider targeted data augmentation or re-sampling techniques (e.g., SMOTE for specific minority *demographic* groups within the defaulting class) to ensure the model learns more robust patterns for these underrepresented or underserved groups. This should be done carefully to avoid reinforcing existing biases or creating new ones.

## Stage 4 Tuning: Fairness Impact

The provided model results include "notes" and "hyperparameters" indicating that tuning techniques were applied to the Random Forest and XGBoost models.

### Random Forest
*   **Technique applied**: `class_weight='balanced'` was applied as a hyperparameter.
*   **Effect on training distribution**: This technique adjusts the loss function during training, giving more weight to the minority class (defaults) to address class imbalance. While it doesn't create synthetic samples like SMOTE, it effectively alters the importance of different examples the model sees. This can shift the model's decision boundary, potentially impacting fairness metrics. In this case, the Random Forest achieved better FPR gaps for 'AGE' (0.0232) and 'MARRIAGE' (0.0645) compared to XGBoost, suggesting that this balancing might have contributed to a more equitable treatment of false positives for these attributes, even though Disparate Impact still fell below the 0.80 threshold. The fairness scores reported are after this technique was applied.

### XGBoost
*   **Technique applied**: `GridSearchCV` was used for hyperparameter tuning, with `scoring=recall` as the primary optimization objective. The best parameters found were `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200}`.
*   **Effect on training distribution**: `GridSearchCV` itself does not alter the training data distribution but systematically searches for hyperparameters that optimize a given metric. By explicitly optimizing for 'recall', the tuning process prioritized the model's ability to correctly identify positive cases (defaults). This can lead to a model that is more aggressive in its predictions, potentially increasing False Positive Rates for certain groups if their score distributions overlap with the decision boundary in a way that exacerbates existing disparities. The fairness scores reported are the outcome of this recall-focused optimization. It is likely that this tuning, while improving recall, may have worsened fairness metrics compared to an untuned baseline, as indicated by XGBoost's higher FPR gaps for 'SEX' and 'AGE' relative to the Neural Network and Random Forest. No baseline fairness scores for an untuned XGBoost model are available for direct comparison.

### Neural Network
*   **Technique applied**: The note "feature importance proxied via |W1| sum across hidden units" describes a method for *interpreting* the trained model's feature importance, not a tuning technique applied during training to influence fairness.
*   **Effect on training distribution**: No tuning technique that altered the training distribution or directly aimed to improve fairness during training was applied to the Neural Network, based on the provided notes. The reported fairness scores reflect the model's inherent behavior given its architecture and standard training.