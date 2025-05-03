#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_namespace_packages, setup

def read_requirements():
    with open("requirements.txt", "r") as file:
        requirements = []
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
        return requirements

with open("README.md", "r") as file:
    long_description = file.read()

# with open("VERSION", "r") as file:
#     version = file.read().strip()

setup(
    name="diffusion-smc",
    # version=version,
    author="Angus Phillips, Michael Hutchinson, Hai-Dang Dau",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="",
    # license="Apache License 2.0",
    keywords="",
    author_email="angus.phillips@stats.ox.ac.uk",
    packages=find_namespace_packages(include=["pdds"]),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
