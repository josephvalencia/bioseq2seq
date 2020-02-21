"""  Attention and normalization modules  """
from bioseq2seq.modules.util_class import Elementwise
from bioseq2seq.modules.gate import context_gate_factory, ContextGate
from bioseq2seq.modules.global_attention import GlobalAttention
from bioseq2seq.modules.copy_generator import CopyGenerator, CopyGeneratorLoss, \
    CopyGeneratorLossCompute
from bioseq2seq.modules.multi_headed_attn import MultiHeadedAttention
from bioseq2seq.modules.embeddings import Embeddings, PositionalEncoding, \
    VecEmbedding
from bioseq2seq.modules.average_attn import AverageAttention

__all__ = ["Elementwise", "context_gate_factory", "ContextGate",
           "GlobalAttention", "CopyGenerator",
           "CopyGeneratorLoss", "CopyGeneratorLossCompute",
           "MultiHeadedAttention", "Embeddings", "PositionalEncoding",
           "AverageAttention", "VecEmbedding"]
