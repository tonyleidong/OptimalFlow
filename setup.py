import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dynapipe", 
    version="0.1.4",
    author="Tony Dong",
    author_email="tonyleidong@gmail.com",
    description="Dynamic Pipeline is a high-level API to help data scientists building models in ensemble way, and automating Machine Learning workflow with simple coding.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tonyleidong/dynapipe",
    keywords = ['auto machine learning', 'features selection', 'model selection','model preprocessing','pipeline'],
    packages=setuptools.find_packages(),
    include_package_data = True,
    install_requires=[
        'pandas',
        'scikit-learn',
        'statsmodels',
        'scipy',
        'joblib',
        'category_encoders',
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',

)

