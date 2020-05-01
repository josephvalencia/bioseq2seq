"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
"""
import torch
import traceback
import datetime
import time
import tqdm
import bioseq2seq.utils
from bioseq2seq.utils.logging import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from bioseq2seq.bin.evaluator import Evaluator
from bioseq2seq.bin.translate import translate

class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`bioseq2seq.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`bioseq2seq.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`bioseq2seq.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`bioseq2seq.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`bioseq2seq.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`bioseq2seq.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,rank,gpus,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1, gpu_verbose_level=0,
                 report_manager=None, with_align=False, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0]):

        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.with_align = with_align
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.rank = rank
        self.num_gpus = gpus
        self.evaluator = Evaluator()

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d"
                            % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1)/(step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000,
              valid_state=None,
              mode=None):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """

        total_stats = bioseq2seq.utils.Statistics()
        report_stats = bioseq2seq.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        begin_time = time.time()

        for i, (batches, normalization) in enumerate(
                self._accum_batches(train_iter)):

            effective_batch_size = sum([x.tgt.shape[1] for x in batches])
            step = self.optim.training_step

            self._maybe_update_dropout(step)

            self._gradient_accumulation(i,
                batches, normalization, total_stats,
                report_stats)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim.learning_rate(),
                report_stats)

            if valid_iter is not None and step % valid_steps == 0 and self.rank == 0:
                
                if mode == "translate" or mode == "combined":
                    print("Structured validation")
                    valid_stats = self.validate_structured(valid_iter,valid_state,moving_average=self.moving_average)
                else:
                    print("Normal validation")
                    valid_stats = self.validate(valid_iter, moving_average=self.moving_average)
                
                valid_stats = self._maybe_gather_stats(valid_stats)

                self._report_step(self.optim.learning_rate(),
                                  step, valid_stats=valid_stats)

                elapsed = time.time() - begin_time
                # print("Elapsed time: {} minutes".format(elapsed/60.))

                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        print("EARLY STOPPING!!Finishing Training")
                        break

            if (self.model_saver is not None
                and (save_checkpoint_steps != 0
                     and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)

        return total_stats

    def validate(self, valid_iter, moving_average=None):

        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" \
                    else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = bioseq2seq.utils.Statistics()

            for batch in tqdm.tqdm(valid_iter):
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                   else (batch.src, None)
                tgt = batch.tgt

                # F-prop through the model.
                outputs, enc_attn, attns = valid_model(src, tgt, src_lengths,
                                             with_align=self.with_align)
                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)

        if moving_average:
            for param_data, param in zip(model_params_data,
                                         self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        return stats

    def validate_structured(self,valid_iter,valid_state,moving_average=None):

        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" \
                    else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
 
            stats = bioseq2seq.utils.Statistics()
            
            for batch in valid_iter:

                src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
                tgt = batch.tgt

                # F-prop through the model.
                outputs, enc_attn, attns  = valid_model(src, tgt, src_lengths,with_align=self.with_align)
                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)
                # Update statistics.
                stats.update(batch_stats)
            
            # Perform beam-search decoding and compute structured metrics
            s = time.time()
            translations,gold,scores = translate(valid_model, *valid_state,beam_size=1,n_best=1)
            e = time.time()
            print("Decoding time: {}".format(e-s))
            
            top_results,top_n_results = self.evaluator.calculate_stats(translations,gold,full_align=True)
            print(top_results)

            stats.update_structured(top_results,top_n_results)

        if moving_average:
            for param_data, param in zip(model_params_data,self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        return stats

    def _gradient_accumulation(self,batch_num,true_batches, normalization, total_stats,
                               report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):

            true_batch_num = batch_num *self.accum_count + k

            msg = "Entering batch {} with src shape {} and tgt shape {} from device {}"
            print(msg.format(true_batch_num,batch.src[0].shape,batch.tgt.shape,self.rank))

            target_size = batch.tgt.size(0)

            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)

            if src_lengths is not None and report_stats is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt

            bptt = False
            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()

                if self.num_gpus > 1:
                    parallel_model = DDP(self.model,device_ids = [self.rank],output_device = self.rank)
                    outputs,enc_attn,attns = parallel_model(src, tgt, src_lengths, bptt=bptt, with_align=self.with_align)
                else:
                    outputs,enc_attn,attns = self.model(src, tgt, src_lengths, bptt=bptt, with_align=self.with_align)
                    print("outputs",outputs.shape)

                bptt = True

                # 3. Compute loss.
                try:
                    loss, batch_stats = self.train_loss(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=0,
                        trunc_start=j,
                        trunc_size=trunc_size)
                    if loss is not None:
                        self.optim.backward(loss)

                    if total_stats is not None and report_stats is not None:
                        total_stats.update(batch_stats)
                        report_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d",
                                self.optim.training_step, k)

                if self.accum_count == 1:
                    self.optim.step()

                # If truncated, don't backprop fully.

                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()


    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
            """
            Gather statistics in multi-processes cases
            Args:
                stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                    or None (it returns None in this case)
            Returns:
                stat: the updated (or unchanged) stat object
            """
            if stat is not None and self.n_gpu > 1:
                return bioseq2seq.utils.Statistics.all_gather_stats(stat)
            return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `bioseq2seq.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `bioseq2seq.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)
