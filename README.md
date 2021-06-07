# ez-prediction
- Sometimes it is desirable to use clinical information like age, gender, clinical index, or high through put experiment data to predict disease outcome, or perform so-called molecular diagnosis. 

- Traditional machine learning methods work well for these proposes. This repo contains scripts for clinical prediction, all implemented as `R` markdown.

- Before you perform such machine learning, you'd better visualization your data with **PCA**, **MDS**, or **hierarchical clustering**, colored with sample labels. 
  - Under some cases, tumor samples and tumor adjacent normal samples for example, different samples could form highly distinct clusters, and supervised learning is expected to have very high accuracy, hence not even necessary. 
  - Supervised learning here may useful for identify mild difference, like tumor samples from patients with good prognosis and bad prognosis, or even more mild difference, like plasma samples from cancer patient and healthy donors.
  - Sample data here is actually some **COAD** tumor and paired tumor adjacent normal tissue data from **TCGA**. As described bellow, they are quiet distinct, and **accuracy on test set should near 100%**. One would **never** perform such analysis in real practice, this data is only used to exemplify how to use some machine learning package in R.
  
- Several R packages is required:

  - [edgeR](https://bioconductor.org/packages/release/bioc/html/edgeR.html): for data normalization, and identify differential genes
  - [caret](https://topepo.github.io/caret/): for dataset splitting and feature selection
  - [pROC](https://cran.r-project.org/web/packages/pROC/index.html): for performance evaluation
  - [glmnet](https://cran.r-project.org/web/packages/glmnet/index.html): for (regularized) logistic regression
  - [e1071](https://cran.r-project.org/web/packages/e1071/index.html): for SVM
  - [randomForest](https://cran.r-project.org/web/packages/randomForest/index.html): for random forest

- Some R package you may interest in:
  - More packages for feature selection
    - [FSinR](https://cran.r-project.org/web/packages/FSinR/index.html)
    - This repo <https://github.com/FrancisArgnR/R-FeatureSelection-Packages> lists many packages for feature selection.


## Prepare input data
### Preprocessing

- See `notebooks/preprocessing.Rmd`
- Normalization is required for RNA-seq data
- For gene expression, better log transform the data
- Scale each feature, make its numeric values close to 0, like mean of 0 and standard deviation of 1.  Such scaling is **required for SVM and Logistic regression, not required for random forest**. See discussion here <https://stackoverflow.com/questions/8961586/do-i-need-to-normalize-or-scale-data-for-randomforest-r-package>
- Ideally, preprocessing should be applied to training and testing set separately, to avoid potential information leaking from testing set to training set.

## Dataset splitting

- See `notebooks/splitting.Rmd`
- If you want samples with same label to evenly distributed in training and testing set, perform stratified splitting 

## Model training 

### Feature selection
- Search for a feature subset of given size, that has good predictive capacity.
- See `notebooks/feature-selection.Rmd`
- Whether feature selection is necessary, and how to perform it?
  - See following discussions
    - <https://stats.stackexchange.com/questions/215154/variable-selection-for-predictive-modeling-really-needed-in-2016>
    - <https://datascience.stackexchange.com/questions/16062/is-feature-selection-necessary>
  - Seems the consensus if sample size is large, feature selection is not necessary.
  - If you want to perform feature selection, make sure only use data in training set. Also, if you use whole training set for feature selection, then use these features for classification, the resulting cross validation performance will be over estimated, and only performance on testing set makes sense.

- Feature selection methods
  - Methods that optimize for single feature
    - Calculate a measure of correlation (p value of some statistical testing, mutual information, AUROC, etc) between each feature and response variable
    - Rank features by such measure, take top features
  - Methods that optimize for multiple features
    - Search for a feature subset that maximize the cross validation performance of a classifier on training data
    - Could (in theory) consider interactions between different features.
    - Impractical to enumerate all feature subset under most cases, have to rely on heuristics like recursive elimination, simulated annealing, etc
  -  It's very simple to implement most univariate feature selection method using native R.
  - [caret](https://topepo.github.io/caret/) package implement multiple feature selection method. See introduction here <https://topepo.github.io/caret/feature-selection-overview.html>.
  - [FSinR](https://cran.r-project.org/web/packages/FSinR/vignettes/FSinR.html) also have multiple implementations.
  - A measure of feature importance is availble for most classifier. You can simply use the most important ones.
  


### Model fitting
- Note that R packages typically distinguish regression tasks and classification tasks by data type of the response variable. This is different from [sklearn](https://scikit-learn.org/), which provide separated API for classification and regression.
- If your response is factor in `R`, it will perform  regression, or if your response is a numeric vector, it will perform regression.
- So **for classification, make sure your input response variable is a vector of R factors**
- Logistic regression, SVM, random forest or gradient boosting?
  - All is OK. 
  - If you emphasis interpretability rather than performance, use logistic regression. 
  - If you want your model to tolerate dirty data (minimal preprocessing), run fast, and expect good performance, use tree-based method (random forest or gradient boosting).
  - SVM is also a good choice under most situation.
  - More advanced gradient boosting method often outperform random forest at more complicate problem if the parameter is well tuned, but for data as simple as cancer-normal binary classification, random forest is enough.
  - Here we shall focus on logistic regression and random forest

####  Logistic Regression

- In context of linear model, logistic regression is a generalize linear model with binomial response and logit link function (logistic function, or sigmoid function). Sometimes it's also called logit model.  If you replace logit link function with probit link function (CDF of standard normal distribution) , it becomes a probit model.
- In context of machine learning, logistic regression actually perform binary classification
- You can use `glm` in base R with `binomial` family (logit is the default link function) to perform logistic regression, if number of feature is small.
- If number of features get larger, you'd better use a regularized version of logistic regression, such as that  implemented in `glmnet` (with binomial family), see introduction here <https://glmnet.stanford.edu/articles/glmnet.html>. Regularization shrink weight of features toward zero to mitigate over fitting. 
- If we use L1 regularization, it's called `LASSO`. If we use L2 regularization, it's called `ridge regression`. If both is used, it's called `elastic net`.
- In `glmnet`, the loss is a elastic net loss, that why this package called `glmnet`. 
- `alpha` controls weighting of L1 and L2 regularization. `alpha`=1 is the `LASSO` penalty, and `alpha`=0 the ridge penalty. By default, `alpha`=1. That means `glmnet` perform `LASSO` by default.
- `lambda` controls degree of regularization. 
- Logistic regression can be generalized to multi-class classification, you can simply replace `binomial` family with `multinomial` family. 

####  Random Forest

- Random forest is a bagging based ensemble method. It samples features and samples to fit multiple decision trees, and makes final prediction by voting.
- Random forest fit each decision tree using data sampled by bootstrap (sampling with replacement). That means for a subset of the trees, some samples are used multiple times, while some samples are never used. So we have a concept called [OOB error](https://en.wikipedia.org/wiki/Out-of-bag_error). You can evaluate performance with out cross validation by directly use OOB error.
- Increasing number of trees will never lead a random forest to over-fit.
- Usually you not need to tune number of trees. See discussion here <https://stats.stackexchange.com/questions/348245/do-we-have-to-tune-the-number-of-trees-in-a-random-forest>. To summarize, "Tuning the number of trees is unnecessary; instead, simply set the number of trees to a large, computationally feasible number".
- `randomForest` package provide a function `tuneRF`. It tunes the hyper-parameter `mtry` base on OOB error. `mtry` is "number of predictors sampled for splitting at each node" when building decision trees. See here <https://stats.stackexchange.com/questions/102867/random-forest-mtry-question> if you get confused with what `mtry` actually means.

#### Parameter tuning

- `notebooks/tune.Rmd`
- Default parameters usually works quite well under most situation.
- If you want to tune parameters
  - K fold cross validation, or leave-one-out cross validation if sample size is very small.
  - You can take advantage of OOB if you use random forest.

#### Performance evaluation
- `notebooks/performance.Rmd`
- For each sample, in binary cases, the model gives P(y_i=1|X_i,Model)
- See <https://en.wikipedia.org/wiki/Confusion_matrix> and <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>
- Also see <https://people.inf.elte.hu/kiss/11dwhdm/roc.pdf>
- Some alias
  - sensitivity, recall, TPR
  - FPR, 1 - specificity
  - precision, PPV
- We shall calculate the following metrics from known labels y_i (binary value in 0,1) and predicted probability P(y_i=1|X_i,Model) 
- To calculate recall and precision, we should specify a predefined cutoff. For different cutoff, we can have different `FPR`, recall and precision. That is to say, every (`FPR`,`TPR`) pair, or (1-specificity,sensitivity) is a point on ROC (Receiver operating characteristic) curve, every  (`recall`,`precision`) pair is a point on precision recall curve (PRC)
- For whole validation set, traverse all possible cutoff, we have a single ROC curve and single PRC curve, hence a single `AUROC` value and a single `AUPRC` value.
- For clinical application, seems `AUROC`  is reported in most publications, sensitivity  and specificity some times is also reported. The confidence interval can also be reported. As there is different (1 - specificity, sensitivity) pair, we often take point closest to up-left corner (`closest.topleft` in pROC), or point where maximize sensitivity+specificity, (`youden` in `pROC`). 













