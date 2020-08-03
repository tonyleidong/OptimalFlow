===============
autoPipe Module
===============

Description : 
 - This module is used for pipeline connection:
    * Connect autoPP, autoFS, and autoCV like scikit-learn pipeline approach.
    * Automated the iteration of cross-experiment to find the best baseline model
    * Generate comparable and parameter-tracable dictionaies and reports to support autoVIZ and autoFlow modules
 
 - Pipeline Components:
    * autoPP - dynaPreprocessing() Class in autoPP module
    * Datasets Splitting - pipeline_splitting_rule() Function in utilis_funs module
    * autoFS - dynaFS_clf() or dynaFS_reg() Class in autoFS module
    * autoCV - dynaClassifier() or dynaRegressor() Class in autoCV module
    * Model Evaluate - evaluate_model() Class in autoCV module