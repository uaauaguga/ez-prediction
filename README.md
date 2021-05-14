# ez-prediction
- Sometimes it is favorable to use clinical information like age, gender, clinical index, or high through put experiment data to predict disease outcome, or perform so-called moleculer diagnosis. 
- Traditional machine learning methods work well for these proposes. This repo contains scripts for clinical prediction, all implemented in `R`.
- Before you perform such machine learning, you'd better visualization your data with PCA, or hierarchical clustering, colored with sample labels. 
  - Under some cases, tumor samples and tumor adjacent normal samples for example, different samples could form highly distinct clusters, and supervised learning is expected to have very high accuracy, hence not even necessary. 
  - Supervised learning here may useful for identify mild difference tumor samples from patients with good prognosis and bad prognosis, or plasma samples from cancer patient and healthy donors.

## Prepare input data
### Preprocessing

- Normalization is required for RNA-seq data
- For gene expression, better log transform the data
- Scale each feature, make its numeric values close to 0, like mean of 0 and standard deviation of 1.  Required for SVM and Logistic regression, not required for random forest and gradient boosting.

## Dataset splitting

- If you want samples with same label to evenly distributed in training and testing set, perform stratified splitting 

## Model training 

### Feature selection
- Whether feature selection is necessary, and how to perform it?
  - See following discussions
    - <https://stats.stackexchange.com/questions/215154/variable-selection-for-predictive-modeling-really-needed-in-2016>
    - <https://datascience.stackexchange.com/questions/16062/is-feature-selection-necessary>
  - Seems the consensus if sample size is large, feature selection is not necessary.
  - If you want to perform feature selection, make sure only use data in training set. Also, if you use whole training set for feature selection, then use these features for classification, the resulting cross validation performance will be over estimated, and only performance on testing set makes sense.

### Model fitting
- Logistic regression, SVM, random forest or gradient boosting?
  - All is OK. 
  - If you emphasis interpretability rather than performance, use logistic regression. 
  - If you want your model to tolerate dirty data, and expect good performance, use tree-based method (random forest and gradient boosting).
  - SVM is also a good choice under most situation.

#### Parameter tuning

- Five fold cross validation, or leave-one-out cross validation if sample size is very small.
- Choose best hyper-parameter with respect to  average AUROC cross different cross validation runs.

#### Performance evaluation
- For each sample, in binary cases, the model gives P(y_i=1|X_i,Model)
- We shall calculate the following metrics from known labels y_i (binary value in 0,1) and predicted probability P(y_i=1|X_i,Model) 
  - FPR
  - Sensitivity / Recall / TPR
  - Precision / Specificity /PPV
  - ROC curve and AUROC
  - PRC curve and AUPRC
- To calculate FPR, recall and precision, we should specify a predefined cutoff. For different cutoff, we can have different FPR, recall and precision. That is to say, every (FPR,TPR) pair is a point on ROC , every  (Precision,Recall) pair is a point on PRC 
- For whole validation set, traverse all possible cutoff, we have a single ROC curve and single PRC curve, hence a single AUROC value and a single AUPRC value.
- For clinical application, seems AUROC  is reported in most publications, sensitivity  and specificity some times is also reported. As there is different (sensitivity,specificity) pair, we often take point closest to up-left corner, or point where a line with slope 1 tangent to ROC curve. The confidence interval can also be reported.












