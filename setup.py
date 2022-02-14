#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["dask", "xarray>=0.14.1", "xgcm>=0.2.0"]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest",
]


setup(
    author="Mattia Almansi",
    author_email="mattia.almansi@noc.ac.uk",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description=(
        "OceanSpy: A Python package to"
        " facilitate ocean model data analysis and visualization"
    ),
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="oceanspy",
    name="oceanspy",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    python_requires=">=3.7,<=3.9",
    url="https://github.com/hainegroup/oceanspy",
    # fmt: off
    version='0.2.0',
    # fmt: on
    zip_safe=False,
)
