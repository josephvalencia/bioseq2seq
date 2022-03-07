""" Report manager utility """
from __future__ import print_function
import time
from datetime import datetime

import bioseq2seq

from bioseq2seq.utils.logging import logger
from bioseq2seq.utils.report_manager import ReportMgrBase
from ray import tune

class RayTuneReportMgr(ReportMgrBase):
    def __init__(self, report_every, start_time=-1.):
        """
        A report manager that writes statistics on standard output as well as

        Args:
            report_every(int): Report status every this many sentences
        """
        super(RayTuneReportMgr, self).__init__(report_every, start_time)

    def _report_training(self, step, num_steps, learning_rate,
                         report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps,
                            learning_rate, self.start_time)

        report_stats = bioseq2seq.utils.Statistics()
        return report_stats

    def _report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log('Train perplexity: %g' % train_stats.ppl())
            self.log('Train accuracy: %g' % train_stats.accuracy())
            #tune.report(train_step=step,train_accuracy=train_stats.accuracy(),train_class_accuracy=train_stats.class_accuracy())
        
        if valid_stats is not None:
            self.log('Validation perplexity: %g' % valid_stats.ppl())
            self.log('Validation accuracy: %g' % valid_stats.accuracy())
            #tune.report(valid_step=step,valid_accuracy=valid_stats.accuracy(),valid_class_accuracy=valid_stats.class_accuracy())
