"""Module defining encoders."""
from bioseq2seq.encoders.encoder import EncoderBase
from bioseq2seq.encoders.transformer import TransformerEncoder
from bioseq2seq.encoders.rnn_encoder import RNNEncoder
from bioseq2seq.encoders.cnn_encoder import CNNEncoder
from bioseq2seq.encoders.mean_encoder import MeanEncoder
from bioseq2seq.encoders.audio_encoder import AudioEncoder
from bioseq2seq.encoders.image_encoder import ImageEncoder


str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "cnn": CNNEncoder,
           "transformer": TransformerEncoder, "img": ImageEncoder,
           "audio": AudioEncoder, "mean": MeanEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc"]
