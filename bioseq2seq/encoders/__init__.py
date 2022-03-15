"""Module defining encoders."""
from bioseq2seq.encoders.encoder import EncoderBase
from bioseq2seq.encoders.transformer import TransformerEncoder
from bioseq2seq.encoders.cnn_encoder import CNNEncoder
from bioseq2seq.encoders.fnet import FourierEncoder

str2enc = {"transformer": TransformerEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "CNNEncoder", "FourierEncoder", "str2enc"]
