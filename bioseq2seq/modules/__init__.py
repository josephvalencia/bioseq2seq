"""  Attention and normalization modules  """
from bioseq2seq.modules.util_class import Elementwise
from bioseq2seq.modules.gate import context_gate_factory, ContextGate
from bioseq2seq.modules.conv_multi_step_attention import ConvMultiStepAttention
from bioseq2seq.modules.multi_headed_attn import MultiHeadedAttention
from bioseq2seq.modules.embeddings import Embeddings, PositionalEncoding, \
    VecEmbedding
from bioseq2seq.modules.weight_norm import WeightNormConv2d
from bioseq2seq.modules.average_attn import AverageAttention
from bioseq2seq.modules.answer_pointer import PointerGenerator, PadDecoder

__all__ = ["Elementwise", "context_gate_factory", "ContextGate",
            "ConvMultiStepAttention",
           "MultiHeadedAttention", "Embeddings", "PositionalEncoding",
           "WeightNormConv2d","AverageAttention", "VecEmbedding",
           "PointerGenerator", "PadDecoder"]
