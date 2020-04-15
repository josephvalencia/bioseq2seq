"""Module defining decoders."""
from bioseq2seq.decoders.decoder import DecoderBase
from bioseq2seq.decoders.transformer import TransformerDecoder


str2dec = {"transformer": TransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder","str2dec"]
