Welcome to Dynamic Pipeline's Documentation!
============================================
.. image:: DynamicPipeline_Logo.png
   :width: 200
   :align: center
   
**Dynamic Pipeline** is a high-level API to help data scientists building models in ensemble way, and automating Machine Learning workflow with simple codes. 
Comparing other popular "AutoML or Automatic Machine Learning" APIs, **Dynamic Pipeline** is designed as a ML workflow optimizer with higher-level API targeting to avoid manual repetitive train-along-evaluate experiments in general pipeline building. 
To achieve that, **Dynamic Pipeline** creates *Pipeline Cluster* to assemble all cross-matching pipelines covering major tasks of Machine Learning workflow, and apply traversal-experiment to search the optimal baseline model.
Besides, by modularizing all key pipeline components in reuseable packages, it allows all components to be custom tunable along with high scalability.

.. image:: DynamicPipeline_Workflow.PNG
   :width: 980

Dynamic Pieline share a few useful properties for data scientists:

      * *Easy & less coding* - High-level APIs to implement *Pipeline Cluster* traversal experiments, and each ML component is highly automated and modularized;

      * *Well ensembled* - Each key component is ensemble of popular algorithms w/ optimal hyperparameters tuning included;
      
      * *Hardly omission* - Using * Pipeline Cluster * technique, to cross-experiment with combined permutated input datasets, feature selection, and model selection;
      
      * *Scalable* - Each module could add new algorithms easily due to its ensemble and reuseable coding design;

      * *Adaptable* - *Pipeline Cluster* traversal experiments makes it easier to adapt unseen datasets with the right pipeline;
      
      * *Custom Modify Welcomed* - Support custom settings to add/remove algorithms or modify hyperparameters for elastic requirements.



.. toctree::
   :maxdepth: 3

   installation
   autoPP
   autoFS
   autoCV
   autoPipe
   demos
   authors
   history
   issues

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
  
