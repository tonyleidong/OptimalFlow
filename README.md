## OptimalFlow
[![PyPI Latest Release](https://img.shields.io/pypi/v/OptimalFlow)](https://pypi.org/project/OptimalFlow/)
[![Github Issues](https://img.shields.io/github/issues/tonyleidong/DynamicPipeline)](https://github.com/tonyleidong/DynamicPipeline/issues)
[![License](https://img.shields.io/github/license/tonyleidong/DynamicPipeline)](https://github.com/tonyleidong/DynamicPipeline/blob/master/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/tonyleidong/DynamicPipeline)](https://github.com/tonyleidong/OptimalFlow)
[![Python Version](https://img.shields.io/pypi/pyversions/OptimalFlow)](https://pypi.org/project/OptimalFlow/)


   
#### Author: [Tony Dong](http://www.linkedin.com/in/lei-tony-dong)

**OptimalFlow** is a high-level API toolkit to help data scientists building models in ensemble way, and automating Machine Learning workflow with simple codes. 

Comparing other popular "AutoML or Automatic Machine Learning" APIs, **OptimalFlow** is designed as an omni-ensembled ML workflow optimizer with higher-level API targeting to avoid manual repetitive train-along-evaluate experiments in general pipeline building. 

To achieve that, **OptimalFlow** applies *Pipeline Cluster Traversal Experiments* algorithm to assemble all cross-matching pipelines covering major tasks of Machine Learning workflow, and apply traversal-experiment to search the optimal model. Besides, by modularizing all key pipeline components in reuseable packages, it allows all components to be custom tunable along with high scalability.

The core concept in **OptimalFlow** is *Pipeline Cluster Traversal Experiments*, which is a theory, first raised by Tony Dong during Genpact 2020 GVector Conference, to optimize and automate Machine Learning Workflow using ensemble pipelines algorithm.

Comparing other automatic or classic machine learning workflow's repetitive experiments using single pipeline, *Pipeline Cluster Traversal Experiments* is more powerful, with larger coverage scope, to find the best model without manual intervention, and also more flexible with elasticity to cope with unseen data due to its ensemble designs in each component.


### Documentation:  https://optimal-flow.readthedocs.io/

### Installation
```python
pip install optimalflow
```

### License:
MIT (C)Tony Dong