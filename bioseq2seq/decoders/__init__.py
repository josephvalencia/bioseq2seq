"""Module defining decoders."""
from bioseq2seq.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from bioseq2seq.decoders.transformer import TransformerDecoder
from bioseq2seq.decoders.cnn_decoder import CNNDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec"]
