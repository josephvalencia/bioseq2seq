from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bioseq2seq',
    description='An implementation of seq2seq for RNA translation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Joseph Valencia",
    author_email="valejose@oregonstate.edu",
    version='1.0.0',
    packages=find_packages(),
) 
