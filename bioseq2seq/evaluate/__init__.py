"""Module defining inputters.
Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from bioseq2seq.evaluate.ensemble import Vote, Prediction, prediction_from_record
from bioseq2seq.evaluate.evaluator import Evaluator

__all__ = ['Vote','Prediction','prediction_from_record','Evaluator']
