""" Main entry point of the bioseq2seq library """
from __future__ import division, print_function

import bioseq2seq.inputters
import bioseq2seq.encoders
import bioseq2seq.decoders
import bioseq2seq.models
import bioseq2seq.utils
import bioseq2seq.modules
import bioseq2seq.bin
from bioseq2seq.trainer import Trainer
import sys
import bioseq2seq.utils.optimizers
import bioseq2seq.translate
bioseq2seq.utils.optimizers.Optim = bioseq2seq.utils.optimizers.Optimizer
sys.modules["bioseq2seq.Optim"] = bioseq2seq.utils.optimizers

# For Flake
__all__ = [bioseq2seq.inputters, bioseq2seq.encoders, bioseq2seq.decoders, bioseq2seq.models,
           bioseq2seq.utils, bioseq2seq.modules, "Trainer"]

__version__ = "1.0.0"
