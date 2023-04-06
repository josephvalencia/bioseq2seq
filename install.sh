#!/usr/bin/env bash
rm -rf build/
rm -rf dist/
rm -rf bioseq2seq.egg-info/
rm -rf venv/lib/python3.7/site-packages/bioseq2seq/
rm -rf venv/lib/python3.7/site-packages/bioseq2seq-1.0.0.dist-info/
python setup.py sdist bdist_wheel
pip install dist/bioseq2seq-1.0.0-py3-none-any.whl --force-reinstall
