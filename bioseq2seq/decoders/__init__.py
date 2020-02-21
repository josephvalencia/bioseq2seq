"""Module defining decoders."""
from bioseq2seq.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from bioseq2seq.decoders.transformer import TransformerDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "transformer": TransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder",
           "InputFeedRNNDecoder", "str2dec"]
