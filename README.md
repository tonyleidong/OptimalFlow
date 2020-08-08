## Dynamic Pipeline
[![PyPI Latest Release](https://img.shields.io/pypi/v/dynapipe)](https://pypi.org/project/dynapipe/)
[![PyPI - Downloads](https://img.shields.io/pypi/dw/dynapipe?style=plastic)](https://pypi.org/project/dynapipe/)
[![Github Issues](https://img.shields.io/github/issues/tonyleidong/DynamicPipeline)](https://github.com/tonyleidong/DynamicPipeline/issues)
[![License](https://img.shields.io/github/license/tonyleidong/DynamicPipeline)](https://github.com/tonyleidong/DynamicPipeline/blob/master/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/tonyleidong/DynamicPipeline)](https://github.com/tonyleidong/dynapipe)
[![Python Version](https://img.shields.io/pypi/pyversions/dynapipe)](https://pypi.org/project/dynapipe/)


   
#### Author: [Tony Dong](http://www.linkedin.com/in/lei-tony-dong)

<img src="https://github.com/tonyleidong/DynamicPipeline/blob/master/docs/DynamicPipeline_Logo.png" width="80">**Dynamic Pipeline** is a high-level API toolkit to help data scientists building models in ensemble way, and automating Machine Learning workflow with simple codes. 

Comparing other popular "AutoML or Automatic Machine Learning" APIs, **Dynamic Pipeline** is designed as an omni-ensembled ML workflow optimizer with higher-level API targeting to avoid manual repetitive train-along-evaluate experiments in general pipeline building. 

To achieve that, **Dynamic Pipeline** applies *Pipeline Cluster Traversal Experiments* algorithm to assemble all cross-matching pipelines covering major tasks of Machine Learning workflow, and apply traversal-experiment to search the optimal baseline model.

Besides, by modularizing all key pipeline components in reuseable packages, it allows all components to be custom tunable along with high scalability.


<img src="https://github.com/tonyleidong/DynamicPipeline/blob/master/docs/DynamicPipeline_Workflow.PNG" width="980">

The core concept in **Dynamic Pipeline** is *Pipeline Cluster Traversal Experiments*, which is a theory, first raised by Tony Dong during Genpact 2020 GVector Conference, to optimize and automate Machine Learning Workflow using ensemble pipelines algorithm.

Comparing other automatic or classic machine learning workflow's repetitive experiments using single pipeline, *Pipeline Cluster Traversal Experiments* is more powerful, with larger coverage scope, to find the best model without manual intervention, and also more flexible with elasticity to cope with unseen data due to its ensemble designs in each component.

<img src="https://github.com/tonyleidong/DynamicPipeline/blob/master/docs/PipelineClusterTraversalExperiments.PNG" width="980">


In summary, **Dynamic Pipeline** shares a few useful properties for data scientists:

* *Easy & less coding* - High-level APIs to implement *Pipeline Cluster Traversal Experiments*, and each ML component is highly automated and modularized;

* *Well ensembled* - Each key component is ensemble of popular algorithms w/ optimal hyperparameters tuning included;
      
* *Hardly omission* - Using *Pipeline Cluster Traversal Experiments*, to cross-experiment with combined permutated input datasets, feature selection, and model selection;
      
* *Scalable* - Each module could add new algorithms easily due to its ensemble and reuseable coding design;

* *Adaptable* - *Pipeline Cluster Traversal Experiments* makes it easier to adapt unseen datasets with the right pipeline;
      
* *Custom Modify Welcomed* - Support custom settings to add/remove algorithms or modify hyperparameters for elastic requirements.

### Documentation:  https://dynamic-pipeline.readthedocs.io/

### Installation
```python
pip install dynapipe
```

### Core Modules: 
 - autoPP for feature preprocessing
 - autoFS for classification/regression features selection
 - autoCV for classification/regression model selection and evaluation
 - autoPipe for *Pipeline Cluster Traversal Experiments* 

### Updates on 8/7/2020
 * Add estimators: HuberRegressor, RidgeCV, LassoCV, SGDRegressor, and HistGradientBoostingRegressor
 * Modify parameters.json, and reset_parameters.json for the added estimators
 * Add autoViz for classification problem model retrieval diagram

### License:
MIT
