"""Module defining various utilities."""
from bioseq2seq.utils.misc import split_corpus, aeq, use_gpu, set_random_seed
from bioseq2seq.utils.alignment import make_batch_align_matrix
from bioseq2seq.utils.report_manager import ReportMgr, build_report_manager
#from bioseq2seq.utils.raytune_utils import RayTuneReportMgr
from bioseq2seq.utils.statistics import Statistics
from bioseq2seq.utils.optimizers import MultipleOptimizer, \
    Optimizer, AdaFactor
from bioseq2seq.utils.earlystopping import EarlyStopping
__all__ = ["split_corpus", "aeq", "use_gpu", "set_random_seed", "ReportMgr",
           "build_report_manager", "Statistics",
           "MultipleOptimizer", "Optimizer", "AdaFactor", "EarlyStopping",
            "make_batch_align_matrix"]
