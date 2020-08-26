## OptimalFlow
[![PyPI Latest Release](https://img.shields.io/pypi/v/OptimalFlow)](https://pypi.org/project/OptimalFlow/)
[![PyPI - Downloads](https://img.shields.io/pypi/dw/OptimalFlow?style=plastic)](https://pypi.org/project/OptimalFlow/)
[![Github Issues](https://img.shields.io/github/issues/tonyleidong/OptimalFlow)](https://github.com/tonyleidong/OptimalFlow/issues)
[![License](https://img.shields.io/github/license/tonyleidong/OptimalFlow)](https://github.com/tonyleidong/OptimalFlow/blob/master/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/tonyleidong/OptimalFlow)](https://github.com/tonyleidong/OptimalFlow)
[![Python Version](https://img.shields.io/pypi/pyversions/OptimalFlow)](https://pypi.org/project/OptimalFlow/)


   
#### Author: [Tony Dong](http://www.linkedin.com/in/lei-tony-dong)

<img src="https://github.com/tonyleidong/OptimalFlow/blob/master/docs/OptimalFlow_Logo.png" width="150">

**OptimalFlow** is an Omni-ensemble Automated Machine Learning toolkit, which is based on *Pipeline Cluster Traversal Experiment* approach, to help data scientists building optimal models in easy way, and automate Machine Learning workflow with simple codes.

Comparing other popular "AutoML or Automated Machine Learning" APIs, **OptimalFlow** is designed as an omni-ensembled ML workflow optimizer with higher-level API targeting to avoid manual repetitive train-along-evaluate experiments in general pipeline building. 

To achieve that, **OptimalFlow** applies *Pipeline Cluster Traversal Experiments* algorithm to assemble all cross-matching pipelines covering major tasks of Machine Learning workflow, and apply traversal-experiment to search the optimal baseline model.

Besides, by modularizing all key pipeline components in reuseable packages, it allows all components to be custom tunable along with high scalability.


<img src="https://github.com/tonyleidong/OptimalFlow/blob/master/docs/OptimalFlow_Workflow.PNG" width="980">

The core concept in **OptimalFlow** is *Pipeline Cluster Traversal Experiments*, which is a theory, first raised by Tony Dong during Genpact 2020 GVector Conference, to optimize and automate Machine Learning Workflow using ensemble pipelines algorithm.

Comparing other automated or classic machine learning workflow's repetitive experiments using single pipeline, *Pipeline Cluster Traversal Experiments* is more powerful, with larger coverage scope, to find the best model without manual intervention, and also more flexible with elasticity to cope with unseen data due to its ensemble designs in each component.

<img src="https://github.com/tonyleidong/OptimalFlow/blob/master/docs/PipelineClusterTraversalExperiments.PNG" width="980">


In summary, **OptimalFlow** shares a few useful properties for data scientists:

* *Easy & less coding* - High-level APIs to implement *Pipeline Cluster Traversal Experiments*, and each ML component is highly automated and modularized;

* *Well ensembled* - Each key component is ensemble of popular algorithms w/ optimal hyperparameters tuning included;
      
* *Omni-Coverage* - Using *Pipeline Cluster Traversal Experiments*, to cross-experiment with combined permutated input datasets, feature selection, and model selection;
      
* *Scalable* - Each module could add new algorithms easily due to its ensemble and reuseable coding design;

* *Adaptable* - *Pipeline Cluster Traversal Experiments* makes it easier to adapt unseen datasets with the right pipeline;
      
* *Custom Modify Welcomed* - Support custom settings to add/remove algorithms or modify hyperparameters for elastic requirements.

### Documentation:  https://Optimal-Flow.readthedocs.io/

### Installation
```python
pip install OptimalFlow
```

### Core Modules: 
 - autoPP for feature preprocessing
 - autoFS for classification/regression features selection
 - autoCV for classification/regression model selection and evaluation
 - autoPipe for *Pipeline Cluster Traversal Experiments* 
 - autoViz for pipeline cluster visualization. Current available: Model retrieval diagram
 - autoFlow for logging & tracking.

### Notebook Demo:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tonyleidong/OptimalFlow/master?filepath=tests%2Fnotebook_demo.ipynb)


#### Updates on 8/26/2020
------------------
* Fix evaluate_model() function's bugs in coping with regression problem
* Add reg_table_report() function to create dynamic table report for regression problem in autoViz

#### Updates on 8/24/2020
* Fix evaluate_model() function's precision_score issue when running modelmulti-class classification problems
* Add custom_selectors args for customized algorithm settings with autoFS's 2 classes(dynaFS_reg, dynaFS_clf)

#### Updates on 8/20/2020
 * Add Dynamic Table for Pipeline Cluster Model Evaluation Report in autoViz module
 * Add custom_estimators args for customized algorithm settings with autoCV's 4 classes(dynaClassifier,dynaRegressor,fastClassifier, and fastRegressor)  

#### Updates on 8/14/2020
 * Add *fastClassifier*, and *fastRegressor* class which are both random parameter search based
 * Modify the display settings when using dynaClassifier in non in_pipeline mode

#### Updates on 8/10/2020
 * Stable 0.1.0 version release on Pypi
 
#### Updates on 8/7/2020
 * Add estimators: HuberRegressor, RidgeCV, LassoCV, SGDRegressor, and HistGradientBoostingRegressor
 * Modify parameters.json, and reset_parameters.json for the added estimators
 * Add autoViz for classification problem model retrieval diagram

#### License:
MIT
