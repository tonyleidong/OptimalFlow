===============
autoPipe Module
===============

Description : 
 - This module is used to build *Pipeline Cluster Traversal Experiments*:
    * Create sequential components of *Pipeline Cluster Traversal Experiments*
    * Apply traversal experiments through pipeline cluster to find the best baseline model
    * Generate comparable and parameter-tracable dictionaies and reports to support autoVIZ and autoFlow modules
 
 - Build Steps:
    * autoPP - dynaPreprocessing() Class in autoPP module
    * Datasets Splitting - pipeline_splitting_rule() Function in utilis_funs module
    * autoFS - dynaFS_clf() or dynaFS_reg() Class in autoFS module
    * autoCV - dynaClassifier() or dynaRegressor() Class in autoCV module
    * Model Evaluate - evaluate_model() Class in autoCV module
    
.. image:: PipelineClusterTraversalExperiments.PNG
   :width: 980
   
autoPipe
---------------------

.. autoclass:: dynapipe.autoPipe.autoPipe
    :members:
