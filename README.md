# ez-prediction
- Sometimes it is desirable to use clinical information like age, gender, clinical index, or high through put experiment data to predict disease outcome, or perform so-called molecular diagnosis. 

- Traditional machine learning methods work well for these proposes. This repo contains scripts for clinical prediction, all implemented as `R` markdown.

- Before you perform such machine learning, you'd better visualization your data with **PCA**, **MDS**, or **hierarchical clustering**, colored with sample labels. 
  - Under some cases, tumor samples and tumor adjacent normal samples for example, different samples could form highly distinct clusters, and supervised learning is expected to have very high accuracy, hence not even necessary. 
  - Supervised learning here may useful for identify mild difference, like tumor samples from patients with good prognosis and bad prognosis, or even more mild difference, like plasma samples from cancer patient and healthy donors.
  - Sample data here is actually some **COAD** tumor and paired tumor adjacent normal tissue data from **TCGA**. As described bellow, they are quiet distinct, and **accuracy on test set should near 100%**. One would **never** perform such analysis in real practice, this data is only used to exemplify how to use some machine learning package in R.
  
- Several R packages is required:

  - [edgeR](https://bioconductor.org/packages/release/bioc/html/edgeR.html): for data normalization, and identify differential genes
  - [caret](https://topepo.github.io/caret/): for dataset splitting
  - [pROC](https://cran.r-project.org/web/packages/pROC/index.html): for performance evaluation
  - [glmnet](https://cran.r-project.org/web/packages/glmnet/index.html): for (regularized) logistic regression
  - [e1071](https://cran.r-project.org/web/packages/e1071/index.html): for SVM
  - [randomForest](https://cran.r-project.org/web/packages/randomForest/index.html): for random forest

## Prepare input data
### Preprocessing

- See `notebooks/preprocessing.Rmd`
- Normalization is required for RNA-seq data
- For gene expression, better log transform the data
- Scale each feature, make its numeric values close to 0, like mean of 0 and standard deviation of 1.  Such scaling is **required for SVM and Logistic regression, not required for random forest and gradient boosting tree**. See discussion here <https://stackoverflow.com/questions/8961586/do-i-need-to-normalize-or-scale-data-for-randomforest-r-package>

## Dataset splitting

- See `notebooks/splitting.Rmd`
- If you want samples with same label to evenly distributed in training and testing set, perform stratified splitting 

## Model training 

### Feature selection
- See `notebooks/feature-selection.Rmd`
- Whether feature selection is necessary, and how to perform it?
  - See following discussions
    - <https://stats.stackexchange.com/questions/215154/variable-selection-for-predictive-modeling-really-needed-in-2016>
    - <https://datascience.stackexchange.com/questions/16062/is-feature-selection-necessary>
  - Seems the consensus if sample size is large, feature selection is not necessary.
  - If you want to perform feature selection, make sure only use data in training set. Also, if you use whole training set for feature selection, then use these features for classification, the resulting cross validation performance will be over estimated, and only performance on testing set makes sense.

### Model fitting
- Note that R packages distinguish regression tasks and classification tasks data type of the response variable. This is different from sklearn, which provide seperated API for classification and regression tasks.
- If your response is factor in `R`, it will perform  regression, or if your response is a numeric vector, it will perform regression
- So for classification, make sure your input response variable is a vector of R factors
- Logistic regression, SVM, random forest or gradient boosting?
  - All is OK. 
  - If you emphasis interpretability rather than performance, use logistic regression. 
  - If you want your model to tolerate dirty data (minimal preprocessing), run fast, and expect good performance, use tree-based method (random forest or gradient boosting).
  - SVM is also a good choice under most situation.

#### Parameter tuning

- `notebooks/tune.Rmd`
- Default parameters usually works quite well under most situation.
- If you want to tune parameters
  - K fold cross validation, or leave-one-out cross validation if sample size is very small.

#### Performance evaluation
- `notebooks/performance.Rmd`
- For each sample, in binary cases, the model gives P(y_i=1|X_i,Model)
- We shall calculate the following metrics from known labels y_i (binary value in 0,1) and predicted probability P(y_i=1|X_i,Model) 
  - `FPR`
  - `Sensitivity` / `Recall` / `TPR`
  - `Precision` / `Specificity` /`PPV`
  - ROC curve and `AUROC`
  - PRC curve and `AUPRC`
- To calculate `FPR`, recall and precision, we should specify a predefined cutoff. For different cutoff, we can have different `FPR`, recall and precision. That is to say, every (`FPR`,`TPR`) pair is a point on ROC , every  (`Precision`,`Recall`) pair is a point on PRC 
- For whole validation set, traverse all possible cutoff, we have a single ROC curve and single PRC curve, hence a single `AUROC` value and a single `AUPRC` value.
- For clinical application, seems `AUROC`  is reported in most publications, sensitivity  and specificity some times is also reported. As there is different (sensitivity,specificity) pair, we often take point closest to up-left corner (`closest.topleft` in pROC), or point where maximize (sensitivity+specificity, or `youden` in `pROC`). The confidence interval can also be reported.












