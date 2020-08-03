Welcome to Dynamic Pipeline's Documentation!
======================================

.. image:: DynamicPipeline_Logo.png
   :width: 200
   :align: center
   
**Dynamic Pipeline** is a high-level API to help data scientists building models in ensemble way, and automating Machine Learning workflow with simple codes. The core advantage Dynamic Pipeline is it automates general scikit-learn-style pipeline with each component ensembled, and avoids repeated manual cross-experiments with iteration of algrithms combination and searching best parameters. By modularizing all key classic pipeline components in reuseable packages, it allows all components to be custom tunable along with high scalability.


.. image:: DynamicPipeline_Workflow.PNG
   :width: 980

Dynamic Pieline share a few useful properties for data scientists:

      * Easy & less coding - High-level APIs that each ML component is highly automated and modularized;

      * Well ensembled - Each key component is ensemble of popular algorithms w/ optimal parameters searching included;
      
      * Automated & dynamic - autoPipe can build a scikit-learn-style pipline workflow to easily achieve automated ML;
      
      * Hardly omission - Create powerful cross-experiments combined permutated input datasets, feature selection, and model selection;
      
      * Algorithms scalable - Each module could add new algorithms easily due to its ensemble coding structure;
      
      * Adaptable - Support custom settings to add/remove algorithms or modify parameters for elastic requirements.

.. automodule:: dynapipe.autoPP
    :members:

.. automodule:: autoPP

.. toctree::
   :maxdepth: 3

   readme
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
  
