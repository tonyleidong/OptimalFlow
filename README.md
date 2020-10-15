## OptimalFlow
[![PyPI Latest Release](https://img.shields.io/pypi/v/OptimalFlow?style=flat-square)](https://pypi.org/project/OptimalFlow/)
[![Github Issues](https://img.shields.io/github/issues/tonyleidong/OptimalFlow?style=flat-square)](https://github.com/tonyleidong/OptimalFlow/issues)
[![License](https://img.shields.io/github/license/tonyleidong/OptimalFlow?style=flat-square)](https://github.com/tonyleidong/OptimalFlow/blob/master/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/tonyleidong/OptimalFlow?style=flat-square)](https://github.com/tonyleidong/OptimalFlow)
[![Python Version](https://img.shields.io/pypi/pyversions/OptimalFlow?style=flat-square)](https://pypi.org/project/OptimalFlow/)
<!---
[![PyPI - Downloads](https://img.shields.io/pypi/dm/optimalflow?style=flat-square)](https://Optimal-Flow.readthedocs.io/)
-->

#### Author: [Tony Dong](http://www.linkedin.com/in/lei-tony-dong)

<img src="https://github.com/tonyleidong/OptimalFlow/blob/master/docs/OptimalFlow_Logo.png" width="150">

**OptimalFlow** is an Omni-ensemble Automated Machine Learning Python toolkit, which is based on *Pipeline Cluster Traversal Experiment* approach, to help data scientists build optimal models in easy way, and automate Supervised Learning workflow with simple codes. 

**OptimalFlow** wraps the Scikit-learn supervised learning framework to automatically create an ensemble of machine learning pipelines(Omni-ensemble Pipeline Cluster) based on algorithms permutation in each framework component. It includes feature engineering methods in its preprocessing module such as missing value imputation, categorical feature encoding, numeric feature standardization, and outlier winsorization. The models inherit algorithms from Scikit-learn and XGBoost estimators for classification and regression problems. And the extendable coding structure supports adding models from external estimator libraries, which distincts **OptimalFlow** ’s scalability out of most of AutoML toolkits.

**OptimalFlow** uses *Pipeline Cluster Traversal Experiments* as the optimizer to build an omni-ensemble workflow for an optimal baseline model searching, including feature preprocessing/selection optimization, hyperparameters tuning, model selection and assessment.

After version 0.1.10, it added a "no-code" Web App as an application demo built on **OptimalFlow**. The web app allows simple click and selection for all of the parameters inside of OptimalFLow, which means users could build end-to-end Automated Machine Learning workflow without coding at all!(Read more details https://optimal-flow.readthedocs.io/en/latest/webapp.html)

![WebApp Demo](https://github.com/tonyleidong/OptimalFlow/blob/master/docs/OptimalFlow-WebApp-slow.gif)

Comparing other popular "AutoML or Automated Machine Learning" APIs, **OptimalFlow** is designed as an omni-ensembled ML workflow optimizer with higher-level API targeting to avoid manual repetitive train-along-evaluate experiments in general pipeline building. 

To achieve that, **OptimalFlow** applies *Pipeline Cluster Traversal Experiments* algorithm to assemble all cross-matching pipelines covering major tasks of Machine Learning workflow, and apply traversal-experiment to search the optimal baseline model.

Besides, by modularizing all key pipeline components in reuseable packages, it allows all components to be custom tunable along with high scalability.


<img src="https://github.com/tonyleidong/OptimalFlow/blob/master/docs/OptimalFlow_Workflow.PNG" width="980">

The core concept in **OptimalFlow** is *Pipeline Cluster Traversal Experiments*, which is a theory, first raised by Tony Dong during Genpact 2020 GVector Conference, to optimize and automate Machine Learning Workflow using ensemble pipelines algorithm.

Comparing other automated or classic machine learning workflow's repetitive experiments using single pipeline, *Pipeline Cluster Traversal Experiments* is more powerful, with larger coverage scope, to find the best model without manual intervention, and also more flexible with elasticity to cope with unseen data due to its ensemble designs in each component.

<img src="https://github.com/tonyleidong/OptimalFlow/blob/master/docs/PipelineClusterTraversalExperiments.PNG" width="980">


In summary, **OptimalFlow** shares a few useful properties for data scientists:

* *Easy coding* - High-level APIs to implement *Pipeline Cluster Traversal Experiments*, and each machine learning component is highly automated and modularized;

* *Easy transformation* - Focus on process automation for locally implementation,which is easy to transfer current operation and meet compliance restriction(i.e. pharmaceutical compliance policy);

* *Easy maintaining* - Wrap machine learning components to well-modularized code packages without miscellaneous parameters inside;

* *Light & swift* - Easy to transplant among projects. Faster w/ convenience to deploy comparing other cloud-based solution(i.e. Microsoft Azure AutoML);

* *Omni ensemble* - OptimalFlow uses *Pipeline Cluster Traversal Experiments* as the optimizer to build an omni-ensemble workflow for an optimal baseline model searching. Easy for data scientists to implement iterated experiments across all ensemble components in the pipeline;
      
* *High Scalability* - Each module could add new algorithms easily due to its ensemble and reusable coding design. And *Pipeline Cluster Traversal Experiments* makes it easier to adapt unseen datasets with the right pipeline;
      
* *Customization* - Support custom settings to add/remove algorithms or modify hyperparameters for elastic requirements.

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


### An End-to-End OptimalFlow Automated Machine Learning Tutorial with Real Projects
* Part 1: https://towardsdatascience.com/end-to-end-optimalflow-automated-machine-learning-tutorial-with-real-projects-formula-e-laps-8b57073a7b50

* Part 2: https://towardsdatascience.com/end-to-end-optimalflow-automated-machine-learning-tutorial-with-real-projects-formula-e-laps-31d810539102

#### Other Stories:

* Ensemble Feature Selection in Machine Learning using OptimalFlow - Easy Way with Simple Code to Select top Features: https://towardsdatascience.com/ensemble-feature-selection-in-machine-learning-by-optimalflow-49f6ee0d52eb

* Ensemble Model Selection & Evaluation in Machine Learning using OptimalFlow - Easy Way with Simple Code to Select the Optimal Model: https://towardsdatascience.com/ensemble-model-selection-evaluation-in-machine-learning-by-optimalflow-9e5126308f12

* Build No-code Automated Machine Learning Model with OptimalFlow Web App: https://towardsdatascience.com/build-no-code-automated-machine-learning-model-with-optimalflow-web-app-8acaad8262b1

### Support OptimalFlow
If you like OptimalFlow, please consider starring or forking it on GitHub and spreading the word!

[![Github Stars](https://img.shields.io/github/stars/tonyleidong/OptimalFlow?style=social)](https://github.com/tonyleidong/OptimalFlow)
[![Github Forks](https://img.shields.io/github/forks/tonyleidong/OptimalFlow?style=social)](https://github.com/tonyleidong/OptimalFlow)

### Please, Avoid Selling this Work as Yours
Voice from the Author: I am glad if you find **OptimalFlow** useful and helpful. Feel free to add it to your project and let more people know how good it is. But please avoid simply changing the name and selling it as your work. That's not why I'm sharing the source code, at all. All copyrights reserved by Tony Dong following MIT license.

### License:
MIT

### Updates History:
-------------------------

#### Updates on 9/29/2020
* Added SearchinSpace settings page in Web App. Users could custom set estimators/regressors' parameters for optimal tuning outputs.
* Modified some layouts of existing pages in Web App.

#### Updates on 9/16/2020
* Created a Web App based on flask framework as OptimalFlow's GUI, to build PCTE Automated Machine Learning by simply clicks without any coding at all!
* Web App included PCTE workflow bulder, LogsViewer, Visualization, Documentation sections.
* Fix the filename issues in autoViz module, and remove auto_open function when generating new html format plots.

#### Updates on 8/31/2020
* Modify autoPP's default_parameters: Remove "None" in  "scaler", modify "sparsity" : [0.50], modify "cols" : [100]
* Modify autoViz clf_table_report()'s coloring settings 
* Fix bugs in autoViz reg_table_report()'s gradient coloring function  

#### Updates on 8/28/2020
* Remove evaluate_model() function's round() bugs in coping with classification problem
* Move out SVM based algorithm from fastClassifier & fastRegressor's default estimators settings
* Move out SVM based algorithm from autoFS class's default selectors settings

#### Updates on 8/26/2020
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

