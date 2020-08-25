=======
History
=======
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