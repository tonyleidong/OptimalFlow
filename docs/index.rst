Welcome to OptimalFlow's Documentation!
============================================
.. image:: OptimalFlow_Logo.png
   :width: 200
   :align: center
   
**OptimalFlow** is a high-level API toolkit to help data scientists building models in ensemble way, and automating Machine Learning workflow with simple codes. 
Comparing other popular "AutoML or Automatic Machine Learning" APIs, **OptimalFlow** is designed as an omni-ensembled ML workflow optimizer with higher-level API targeting to avoid manual repetitive train-along-evaluate experiments in general pipeline building. 

To achieve that, **OptimalFlow** creates *Pipeline Cluster Traversal Experiments* to assemble all cross-matching pipelines covering major tasks of Machine Learning workflow, and apply traversal-experiment to search the optimal baseline model.
Besides, by modularizing all key pipeline components in reusable packages, it allows all components to be custom tunable along with high scalability.

.. image:: OptimalFlow_Workflow.PNG
   :width: 980

The core concept in **OptimalFlow** is *Pipeline Cluster Traversal Experiments*, which is a theory, first raised by Tony Dong during Genpact 2020 GVector Conference, to optimize and automate Machine Learning Workflow using ensemble pipelines algorithm.
   
Comparing other automatic or classic machine learning workflow's repetitive experiments using single pipeline, *Pipeline Cluster Traversal Experiments* is more powerful, with larger coverage scope, to find the best model without manual intervention, and also more flexible with elasticity to cope with unseen data due to its ensemble designs in each component.  

.. image:: PipelineClusterTraversalExperiments.PNG
   :width: 980
   
   

In summary, **OptimalFlow** shares a few useful properties for data scientists:

      * *Easy & less coding* - High-level APIs to implement *Pipeline Cluster Traversal Experiments*, and each ML component is highly automated and modularized;

      * *Well ensembled* - Each key component is ensemble of popular algorithms w/ optimal hyperparameters tuning included;
      
      * *Hardly omission* - *Pipeline Cluster Traversal Experiments* are designed to cross-experiment with combined permuted input datasets, feature selection, and model selection;
      
      * *Scalable & Consistency* - Each module could add new algorithms easily due to its ensemble & reusable design; no extra needs to modify existing codes for new experiment

      * *Adaptable* - *Pipeline Cluster Traversal Experiments* makes it easier to adapt unseen datasets with the right pipeline;
      
      * *Custom Modify Welcomed* - Support custom settings to add/remove algorithms or modify hyperparameters for elastic requirements.



.. toctree::
   :maxdepth: 3

   installation
   autoPP
   autoFS
   autoCV
   autoPipe
   autoViz
   autoFlow
   demos
   authors
   history
   issues

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
  
