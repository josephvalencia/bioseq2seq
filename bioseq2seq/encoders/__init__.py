"""Module defining encoders."""
from bioseq2seq.encoders.encoder import EncoderBase
from bioseq2seq.encoders.transformer import TransformerEncoder
from bioseq2seq.encoders.rnn_encoder import RNNEncoder


str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder,
           "transformer": TransformerEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "str2enc"]
