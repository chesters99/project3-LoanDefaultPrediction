# project3-LoanDefaultPrediction

## Method
In summary, the overall approach to this project was to firstly to review course materials and Kaggle for
information to support Exploratory Data Analysis (EDA) and modelling. A detailed EDA was then
performed to obtain a good understanding of the available data, then a model selection process was run on a
subset of the data to establish likely high performance models, and finally three selected models were tuned
on various subsets of the data to obtain the lowest log-loss scores.

The technology used for this project was Jupyter Notebooks, Python, with Scikit-learn and XGBoost (0.81)
libraries. The modeling was run on an iMac quad-core i7 4.2GHz, 40GB RAM, and the runtime of the
Python code is approximately 80 minutes. The metric used for this report is log-loss as required.

###Data Exploration and Feature Engineering
Each data field was explored for patterns, null values, anomalies, cross-correlations and relationship to the
loan default rate. Log transforms were tried on the skewed numeric fields but this was of minimal or no
benefit to so this idea was dropped. Consolidation of values in the ‘emp_title’, ‘title’ and ‘zip_code’ fields
was attempted but also resulted in no performance improvement, and so since these fields contained many
values with only a very weak correlation with loan default rates, they were also dropped.

Correlations were then checked, and Grade and ‘fico_range_high’ were dropped as they were very highly
correlated with sub-grade and ‘fico_range_low’ respectively. ‘Loan_amount’ was 0.95 correlated with
installment, but both were retained as there may still be some information available from both. Several fields
(‘dti’, ‘revol_util’, ‘mort_acc’, ‘pub_rec_bankruptcies’) contained NA values, and after careful checking,
these NA’s were replaced with zeros. The ‘earliest_cr_line’ field was converted to an integer offset from
Jan-1950 to aid modelling.

###Algorithms
A broad range of algorithms were applied (with basic tuning) using a 10% sample of the train/test data from
each of the three folds provided, to determine their potential – results shown in Figure 1 below. Clearly logloss
performance was the main criteria, but run-time was also important due to the relatively high volume of
data. Some classifiers were near or even below the log-loss levels required, even with only basic tuning.

K-Nearest Neighbors performance was not as good as Logistic Regression and was almost 100 times slower,
so was dropped. Both linear and RBF kernel-based SVM models run-time was much too long to even appear
in the preliminary results. The Decision Tree classifier and Random Forest performance was middle-ground,
though Random forest was very slow. Adaboost was the worst performer, and Gaussian Naïve Bayes was
also not a good performer. Both Quadratic Discriminant Analysis and Linear Discriminant Analysis reported
warnings regarding collinear columns and were relatively slow.

The three models chosen were Logistic Regression for its simplicity, good performance, minimal tuning, and
very short run-time. Scikit-Learn’s Gradient Boosted Classifier (GBM), and the XGBoost classifier were
also chosen for their excellent performance, despite the relatively long run times.

### Hyper-Parameter Tuning
Tuning the logistic regression model could be done on an entire data fold due to its fast performance and
limited tuning parameters (penalty=’l1’ or ‘l2’, and C).

However tuning GBM and XGBoost was performed initially on only 10% of the available data (due to long
run-times) to get ball-park estimates for the best hyper-parameters. This initial tuning used grid search with 5
fold cross-validation (CV) on learning rate and number of estimators, then on maximum tree depth and
minimum samples for a leaf, then on gamma, then on subsample and colsample by tree, and finally on the
lambda and alpha regularization parameters. The parameters for GBM for tuning were slightly different to
the above, but the same process was followed.

These parameters resulted in models that comfortably beat the baseline required, however it was possible to
optimize further by running a randomized search on ranges of parameters around the best ones found as
described above, and on the first full test fold. This search ran for approximately 18 hours, and resulted in a
small but significant improvement across all three folds.

## Results
Once the tuning was complete, the logistic regression, GBM and XGBoost classifiers were run on the 3 folds
provided in the assessment instructions with the results shown in Figure 3 below. The hyper-parameters for
these final models are included in the Appendix.

## Conclusion
The performance of the simplest model, Logistic Regression, was not quite as good as the two Gradient
Boosted models, but it performed exceptionally well considering its simplicity and very short run-time.
Somewhat surprisingly, it came close to achieving a lower mean log-loss (0.4559) than required.

The best performing models were XGBoost and GBM, with XGBoost having the best performance at a
mean log-loss of 0.4396 over the 3 folds. It should be noted that performance results only slightly worse than
those achieved could be obtained with model run-times less than a quarter of what these models took.
