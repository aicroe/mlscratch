from setuptools import setup, find_packages

setup(
    name="MlBatch",
    version="0.1.0",
    # url=None,
    # project_urls=None,
    author="Diego Garcia",
    author_email="qtimpot@gmail.com",
    description="A tool to run, train and test multiple ML models",
    keywords=["Machine learning", "Deep learning", "Neural network", "Batch", "Train"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=False,
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        # "tensorflow",
    ],
    extras_require={
        "dev": [
            "pylint",
        ],
    },
    zip_safe=True,
    test_suite="tests",
)
