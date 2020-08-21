import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optimalflow", 
    version="0.1.3",
    author="Tony Dong",
    author_email="tonyleidong@gmail.com",
    description="OptimalFlow is an Omni-ensemble Automated Machine Learning toolkit to help data scientists building optimal models in easy way, and automate Machine Learning workflow with simple code.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tonyleidong/OptimalFlow",
    keywords = ['auto machine learning', 'features selection', 'model selection','model preprocessing','pipeline','optimal model'],
    packages=setuptools.find_packages(),
    include_package_data = True,
    install_requires=[
        'pandas',
        'scikit-learn',
        'statsmodels',
        'scipy',
        'joblib',
        'category_encoders',
        'plotly'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',

)

