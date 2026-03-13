from setuptools import find_packages, setup
import os


def parse_requirements(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    with open(filename, "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-e")
        ]


with open("README.md", "r", encoding="utf-8", errors="ignore") as fh:
    long_description = fh.read()

version = {}
with open("ai_data_science_team/_version.py", encoding="utf-8") as fp:
    exec(fp.read(), version)


setup(
    name="Deep-ai-agent",
    version=version["__version__"],
    description="Build and run an AI-powered data science team.",
    author="thanyapisit Buaprakhong",
    author_email="thanyapisit.bua@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deepcoding-agent/ml-datascience",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "machine_learning": ["h2o", "mlflow"],
        "data_science": ["pytimetk", "missingno", "sweetviz"],
        "all": ["h2o", "mlflow", "pytimetk", "missingno", "sweetviz"],
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
