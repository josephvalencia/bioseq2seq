"""Module defining encoders."""
from bioseq2seq.encoders.encoder import EncoderBase
from bioseq2seq.encoders.transformer import TransformerEncoder


str2enc = {"transformer": TransformerEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "str2enc"]
