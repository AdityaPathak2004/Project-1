from setuptools import setup, find_packages

setup(
    name="dskit",
    version="0.1.0",
    author="Aditya Pathak",
    description="A comprehensive data science toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/adityapathak2004/project-1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
    ],
    extras_require={
        "full": [
            "streamlit>=1.28.0",
            "xgboost>=2.0.0",
            "lightgbm>=4.1.0",
            "statsmodels>=0.14.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
