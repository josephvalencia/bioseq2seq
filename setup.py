#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='RiboSeq2Seq',
    description='An implementation of seq2seq for RNA translation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='1.0.0.rc2',
    packages=find_packages(),
    install_requires=[
        "six",
        "tqdm~=4.30.0",
        "torch>=1.2",
        "torchtext==0.4.0",
        "future",
        "configargparse",
        "tensorboard>=1.14",
        "flask",
        "pyonmttok==1.*;platform_system=='Linux'",
    ],
) 
