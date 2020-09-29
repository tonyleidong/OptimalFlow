=======
History
=======

0.1.11 (2020-09-29)
------------------
* Added SearchinSpace settings page in Web App. Users could custom set estimators/regressors' parameters for optimal tuning outputs.
* Modified some layouts of existing pages in Web App.

0.1.10 (2020-09-16)
------------------
* Created a Web App, based on flask framework, as OptimalFlow's GUI. Users could build Automated Machine Learning workflow all clicks, without any coding at all!
* Web App included PCTE workflow bulder, LogsViewer, Visualization, Documentation sections.
* Fix the filename issues in autoViz module, and remove auto_open function when generating new html format plots.

0.1.7 (2020-08-31)
------------------
* Modify autoPP's default_parameters: Remove "None" in  "scaler", modify "sparsity" : [0.50], modify "cols" : [100]
* Modify autoViz clf_table_report()'s coloring settings 
* Fix bugs in autoViz reg_table_report()'s gradient coloring function

0.1.6 (2020-08-28)
------------------
* Remove evaluate_model() function's round() bugs in coping with classification problem
* Move out SVM based algorithm from fastClassifier & fastRegressor's default estimators settings
* Move out SVM based algorithm from autoFS class's default selectors settings

0.1.5 (2020-08-26)
------------------
* Fix evaluate_model() function's bugs in coping with regression problem
* Add reg_table_report() function to create dynamic table report for regression problem in autoViz

0.1.4 (2020-08-24)
------------------
* Fix evaluate_model() function's precision_score issue when running modelmulti-class classification problems
* Add custom_selectors args for customized algorithm settings with autoFS's 2 classes(dynaFS_reg, dynaFS_clf)

0.1.3 (2020-08-20)
------------------
* Add Dynamic Table for Pipeline Cluster Model Evaluation Report in autoViz module
* Add custom_estimators args for customized algorithm settings with autoCV's 4 classes(dynaClassifier,dynaRegressor,fastClassifier, and fastRegressor)  

0.1.2 (2020-08-14)
------------------

* Add *fastClassifier*, and *fastRegressor* class which are both random parameter search based
* Modify the display settings when using dynaClassifier in non in_pipeline mode

0.1.1 (2020-08-10)
------------------

* Add classifiers: LinearSVC, HistGradientBoostingClassifier, SGDClassifier, RidgeClassifierCV.
* Modify Readme.md file.

0.1.0 (2020-08-10)
------------------

* First release on PyPI.