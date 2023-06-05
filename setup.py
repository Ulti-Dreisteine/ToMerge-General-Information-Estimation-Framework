from setuptools import setup, find_packages
from pathlib import Path
import os

VERSION = "0.0.3"
DESCRIPTION = "Package for information estimation, independence test, etc."
this_directory = Path(__file__).parent
long_description = (this_directory/"README.md").read_text(encoding="utf-8")

setup(
    name="giefstat",
    version=VERSION,
    author="Dreisteine",
    author_email="dreisteine@163.com",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=["scikit-learn==0.24.0", "minepy==1.2.6"],
    keywords=["python", "information estimation"],
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ulti-Dreisteine/general-information-estimation-framework",
)