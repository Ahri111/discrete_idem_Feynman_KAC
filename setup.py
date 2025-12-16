from setuptools import setup, find_packages

setup(
    name="discrete_idem",
    version="0.1.0",
    packages=find_packages(),
    
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
    ],
    
    python_requires=">=3.10",
)