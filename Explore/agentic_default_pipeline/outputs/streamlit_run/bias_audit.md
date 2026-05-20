## Bias Detection Summary
All three models exhibit critical bias risks, including direct reliance on protected demographic features like AGE and EDUCATION, and significant use of proxy variables such as LIMIT_BAL and BILL_AMT columns. Given these findings, all models should be blocked from deployment pending further investigation and comprehensive remediation of the identified biases.

## Direct Demographic Feature Audit

### SEX
- SEX does not appear in any model's top-10 feature importances.
- Males default at 24.16% vs Females at 20.79% — a known 3.37 percentage point gap in the dataset.
- The false positive rate must be computed separately for Male and Female clients before deployment. Subgroup false positive rate data is not available in this report, therefore the gap cannot be computed. This is a critical missing piece of information for assessing gender bias.

### AGE — 71-80 Group
- ⚠️ CRITICAL BIAS FLAG: The 71-80 age group is critically underrepresented in the dataset with a count of 15 records and has historically shown a false positive rate of 66.7% in prior analysis.
- The random_forest model uses AGE as its 10th most important feature with an importance of 0.0418036869589222.
- Any model must be tested for whether it performs at chance level (approximately 50% accuracy) for this group specifically. This report does not contain accuracy metrics for the 71-80 age group, which is required before deployment.
- Flagging that deploying a model that performs at chance level for elderly clients constitutes age discrimination in lending.

### EDUCATION — Undocumented Values
- The EDUCATION column in the provided dataset-level signals only contains documented values (1, 2, 3, 4). No undocumented values (0, 5, 6) were found in the dataset-level report (`education_undocumented.count` is 0).
- ⚠️ DIRECT BIAS FLAG: EDUCATION ranks 6 for neural_network. Education is a protected characteristic in lending contexts. The neural_network model uses EDUCATION as its 6th most important feature with an importance of 9.096397874976773.
- If undocumented EDUCATION values were present in the training data, any model prediction driven by them would be uninterpretable and therefore non-compliant.

### MARRIAGE
- Married clients default more than Single clients (approximately 23.46% vs 20.95% in the dataset).
- MARRIAGE does not appear in any model's top-10 feature importances.
- The random_forest model used `class_weight: "balanced"` during training. Optimisation techniques such as `class_weight="balanced"` can increase a model's reliance on demographic features like MARRIAGE even while improving overall accuracy. This report does not provide feature importances before and after optimization for direct comparison, so this pattern cannot be definitively detected here.
- Increased reliance on marital status after optimisation must be investigated before the optimised model is preferred over the baseline.

## Proxy Feature Audit

### LIMIT_BAL
- ⚠️ PROXY BIAS FLAG: LIMIT_BAL is a proxy bias candidate for SEX, AGE, EDUCATION and MARRIAGE.
- The xgboost model uses LIMIT_BAL as its 10th most important feature with an importance of 0.014941778965294361.
- The average LIMIT_BAL must be computed separately for each demographic group.
    - ⚠️ High School (EDUCATION=3) clients have an average LIMIT_BAL of 126099.22, which is -24.39% below the overall average of 166765.55. This is flagged as a potential victim of historical credit discrimination.
    - ⚠️ Undocumented (MARRIAGE=0) clients have an average LIMIT_BAL of 132962.96, which is -20.27% below the overall average of 166765.55. This is flagged as a potential victim of historical credit discrimination.
    - ⚠️ Others (MARRIAGE=3) clients have an average LIMIT_BAL of 98080.5, which is -41.19% below the overall average of 166765.55. This is flagged as a potential victim of historical credit discrimination.
- SHAP values are not available in this report, so whether LIMIT_BAL's contribution differs significantly across demographic groups cannot be checked.

### BILL_AMT Columns
- ⚠️ PROXY BIAS FLAG: BILL_AMT1 through BILL_AMT6 are proxy bias candidates.
- The random_forest model uses BILL_AMT1 as its 8th most important feature with an importance of 0.04219296868339849.
- The distribution of BILL_AMT values must be checked across EDUCATION, SEX, AGE and MARRIAGE groups.
    - ⚠️ Graduate School (EDUCATION=1) clients have a median average bill of 15333.33, which differs by -27.39% from the overall median of 21118.5. This exceeds the 15% threshold.
    - ⚠️ University (EDUCATION=2) clients have a median average bill of 25067.17, which differs by 18.7% from the overall median of 21118.5. This exceeds the 15% threshold.
    - ⚠️ Others (EDUCATION=4) clients have a median average bill of 27567.92, which differs by 30.54% from the overall median of 21118.5. This exceeds the 15% threshold.
    - ⚠️ Undocumented (MARRIAGE=0) clients have a median average bill of 12316.58, which differs by -41.68% from the overall median of 21118.5. This exceeds the 15% threshold.
- Negative BILL_AMT values (credit returns or overpayments) were found in approximately 6.44% of clients (1930 records) and these clients default at a lower rate (0.1648 vs 0.2251). This report does not contain data on whether this subgroup is distributed unevenly across demographic groups, which is required for a complete bias assessment.

### PAY_AMT Columns
- ⚠️ PROXY BIAS FLAG: Payment amounts reflect financial capacity which correlates with socioeconomic status and indirectly with protected characteristics.
- The xgboost model uses PAY_AMT2 (rank 6, importance 0.020006753504276276), PAY_AMT4 (rank 8, importance 0.01684311032295227), and PAY_AMT1 (rank 9, importance 0.015807295218110085) in its top-10 feature importances.
- The neural_network model uses PAY_AMT2 (rank 4, importance 9.27479814968339), PAY_AMT1 (rank 7, importance 9.078555961065883), and PAY_AMT6 (rank 9, importance 8.953903449525459) in its top-10 feature importances.
- A client paying the minimum each month may be doing so by financial necessity — the model cannot distinguish constrained minimum payers from strategic revolvers.
- This report does not contain data on whether clients in the lowest LIMIT_BAL quartile are disproportionately represented in the high false positive group. If the false positive rate for low-limit clients exceeds the overall false positive rate by more than 5 percentage points, it would be evidence of compounded proxy bias. This information is required for a complete assessment.

## Intersectional Bias Check
- ⚠️ Elderly clients (71-80) with low LIMIT_BAL: This combination represents a compounded bias risk that is more serious than either feature alone. Representation bias from too few elderly records (15 clients in 71-80 age group) is likely compounded by proxy bias from LIMIT_BAL, especially if elderly clients were historically assigned lower limits.
- ⚠️ Low-education clients with low LIMIT_BAL: This combination represents a compounded bias risk that is more serious than either feature alone. High School (EDUCATION=3) clients already show a -24.39% gap in average LIMIT_BAL compared to the overall average, indicating potential historical discrimination. This reinforces the proxy signals from both features.
- ⚠️ Female clients with low LIMIT_BAL: This combination represents a compounded bias risk that is more serious than either feature alone. If women were historically assigned lower limits, LIMIT_BAL encodes gender discrimination even when SEX is removed from the model. This report does not contain average LIMIT_BAL by SEX and LIMIT_BAL quartile to confirm this specific pattern, but the risk is plausible and must be investigated.

## Counterfactual Bias Test
Propose the following test to be run before deployment:
- Select 100 clients predicted to default.
- Flip their SEX value from Male to Female or vice versa while keeping all financial features identical.
- If more than 10% of predictions change, the model is directly using gender as a decision factor and must be flagged as discriminatory.
This test must also be run for MARRIAGE and EDUCATION to assess the model's sensitivity to demographic attributes.

## Priority Flags

1.  **CRITICAL — blocks deployment immediately**
    *   ⚠️ AGE — 71-80 Group: Critically underrepresented in the dataset (15 records) and historically high false positive rate (66.7%) in prior analysis. Model performance (accuracy) for this group is not available in this report and is required. Deploying a model that performs at chance level for elderly clients constitutes age discrimination.
    *   ⚠️ SEX: Subgroup false positive rates for Male and Female clients are not available in this report and must be computed before deployment. This is a fundamental requirement for assessing gender bias.
    *   ⚠️ EDUCATION: The neural_network model uses EDUCATION as its 6th most important feature with an importance of 9.096397874976773. This is a direct reliance on a protected characteristic.
    *   ⚠️ Intersectional Bias: Elderly clients (71-80) with low LIMIT_BAL, Low-education clients with low LIMIT_BAL, and Female clients with low LIMIT_BAL represent compounded bias risks that are more serious than individual features. These require specific investigation and remediation.

2.  **HIGH — requires remediation before deployment**
    *   ⚠️ PROXY BIAS FLAG: LIMIT_BAL is a top-10 feature for the xgboost model (rank 10, importance 0.014941778965294361). Significant disparities in average LIMIT_BAL exist for High School (-24.39%), Undocumented Marriage (-20.27%), and Others Marriage (-41.19%) clients, indicating potential historical credit discrimination. This proxy bias must be remediated.
    *   ⚠️ PROXY BIAS FLAG: BILL_AMT1 is a top-10 feature for the random_forest model (rank 8, importance 0.04219296868339849). Significant disparities in median average bill amounts exist for Graduate School (-27.39%), University (18.7%), Others Education (30.54%), and Undocumented Marriage (-41.68%) clients. This proxy bias must be remediated.
    *   ⚠️ PROXY BIAS FLAG: PAY_AMT columns are top-10 features for xgboost (PAY_AMT2 rank 6, PAY_AMT4 rank 8, PAY_AMT1 rank 9) and neural_network (PAY_AMT2 rank 4, PAY_AMT1 rank 7, PAY_AMT6 rank 9). These are strong proxy candidates for socioeconomic status and require investigation and remediation.
    *   ⚠️ random_forest: The use of `class_weight="balanced"` may have increased the model's reliance on demographic features or their proxies. This needs to be investigated for potential disparate impact.

3.  **MEDIUM — requires monitoring after deployment**
    *   ⚠️ random_forest: AGE is a top-10 feature (rank 10, importance 0.0418036869589222). While not a direct block, its presence warrants close monitoring for age-related disparate impact, especially given the known issues with the 71-80 age group.

4.  **LOW — document and review at next model update**
    *   No specific low flags identified that are not already covered by higher severity flags.

## Bias Verdict

**random_forest:** Not deployment ready.
Specific bias remediation steps required:
1.  Compute and analyze subgroup false positive rates for Male and Female clients.
2.  Test model performance (accuracy) for the 71-80 age group.
3.  Investigate the impact of `class_weight="balanced"` on reliance on demographic features and their proxies.
4.  Remediate proxy bias from BILL_AMT1, addressing disparities across EDUCATION and MARRIAGE groups.
5.  Run counterfactual tests for SEX, MARRIAGE, and EDUCATION.

**xgboost:** Not deployment ready.
Specific bias remediation steps required:
1.  Compute and analyze subgroup false positive rates for Male and Female clients.
2.  Test model performance (accuracy) for the 71-80 age group.
3.  Remediate proxy bias from LIMIT_BAL, addressing disparities across EDUCATION and MARRIAGE groups.
4.  Remediate proxy bias from PAY_AMT columns.
5.  Run counterfactual tests for SEX, MARRIAGE, and EDUCATION.

**neural_network:** Not deployment ready.
Specific bias remediation steps required:
1.  Compute and analyze subgroup false positive rates for Male and Female clients.
2.  Test model performance (accuracy) for the 71-80 age group.
3.  Investigate and remediate direct reliance on EDUCATION as a top-6 feature.
4.  Remediate proxy bias from PAY_AMT columns.
5.  Run counterfactual tests for SEX, MARRIAGE, and EDUCATION.

All models are not deployment ready based on the identified bias findings and require significant remediation and further testing before deployment can be considered. No model should be rejected outright based on current findings, but all require substantial work to address the identified biases.

## Stage 4 Tuning: Bias Implications

### random_forest
- **Technique applied**: Class Weight Balancing, specifically `class_weight: "balanced"`.
- **Demographic reliance risk**: ⚠️ Class Weight Balancing reweights training loss to penalise minority-class errors more heavily. If minority-class clients share demographic characteristics, this can amplify those patterns and cause the model to rely more heavily on demographic proxies. AGE ranks 10th in feature importance for this model with an importance of 0.0418036869589222, indicating a potential increased reliance on this demographic feature. This needs further investigation.
- **Threshold tuning and disparate impact**: Not applicable, threshold tuning was not explicitly applied.
- **Bias verdict adjustment**: Indeterminate. While class weighting aims to improve recall for the minority class, its impact on demographic reliance and subgroup fairness cannot be fully assessed without pre-tuning feature importances and subgroup metrics.

### xgboost
- **Technique applied**: GridSearchCV with `scoring=recall`. The best parameters found were `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200}`.
- **Demographic reliance risk**: GridSearchCV itself is a hyperparameter optimization technique and does not directly introduce demographic reliance risk in the same way as resampling or reweighting. However, the chosen hyperparameters might lead to a model that implicitly relies on proxy features. LIMIT_BAL is ranked 10th in feature importance for this model with an importance of 0.014941778965294361, indicating reliance on a proxy feature.
- **Threshold tuning and disparate impact**: Not applicable, threshold tuning was not explicitly applied.
- **Bias verdict adjustment**: Indeterminate. While GridSearchCV aimed to optimize recall, its specific impact on subgroup fairness and reliance on demographic/proxy features cannot be fully assessed without pre-tuning feature importances and subgroup metrics.

### neural_network
- **Technique applied**: No specific bias-related tuning technique (like SMOTE, Class Weight Balancing, or Threshold Tuning) was applied. The `notes` field indicates "feature importance proxied via |W1| sum across hidden units", which describes the method of feature importance calculation, not a tuning technique.
- **Demographic reliance risk**: Not directly applicable as no specific bias-amplifying technique was applied. However, the model directly uses EDUCATION as its 6th most important feature with an importance of 9.096397874976773, indicating a direct reliance on a demographic feature.
- **Threshold tuning and disparate impact**: Not applicable, threshold tuning was not explicitly applied.
- **Bias verdict adjustment**: Indeterminate. The model's direct reliance on EDUCATION is a significant concern, and the absence of specific bias-mitigation tuning means this bias has not been addressed.