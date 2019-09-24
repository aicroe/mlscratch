from setuptools import setup, find_packages

setup(
    name="MlScratch",
    version="0.1.0",
    url="https://github.com/aicroe/mlscratch",
    project_urls={
        "Bug Tracker": "https://github.com/aicroe/mlscratch/issues",
        "Documentation": "https://github.com/aicroe/mlscratch",
        "Source Code": "https://github.com/aicroe/mlscratch",
    },
    author="Diego Garcia",
    author_email="qtimpot@gmail.com",
    description="An abstraction to run, train and test machine learning models",
    keywords=["machine learning", "deep learning", "neural network", "training"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=False,
    python_requires=">=3.6",
    install_requires=[
        "numpy",
    ],
    extras_require={
        "dev": [
            "pylint",
        ],
    },
    zip_safe=True,
    test_suite="tests",
)
