"""Module defining decoders."""
from bioseq2seq.decoders.decoder import DecoderBase
from bioseq2seq.decoders.transformer import TransformerDecoder
from bioseq2seq.decoders.cnn_decoder import CNNDecoder

str2dec = {"transformer": TransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "CNNDecoder","str2dec"]
