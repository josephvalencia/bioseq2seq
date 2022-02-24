#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import time
from itertools import count, zip_longest

import torch
import seaborn as sns
import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import signal
from numpy.fft import rfft

import bioseq2seq.inputters as inputters
from bioseq2seq.translate.beam_search import BeamSearch
from bioseq2seq.translate.greedy_search import GreedySearch
from bioseq2seq.translate.translation import Translation, TranslationBuilder
from bioseq2seq.utils.misc import tile, set_random_seed, report_matrix
from bioseq2seq.utils.alignment import extract_alignment, build_align_pharaoh
from bioseq2seq.translate.attention_stats import SelfAttentionDistribution, EncoderDecoderAttentionDistribution

def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch  # this is a hack
    # Reset current longest leng th at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        # max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    src_elements = count * max_src_in_batch
    return src_elements

class Translator(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (bioseq2seq.modules.NMTModel): NMT model to use for translation
        fields (dict[str, torchtext.data.Field]): A dict
            mapping each side to its list of name-Field pairs.

        src_reader (bioseq2seq.inputters.DataReaderBase): Source reader.
        tgt_reader (bioseq2seq.inputters.TextDataReader): Target reader.
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`bioseq2seq.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`bioseq2seq.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`bioseq2seq.translate.greedy_search.GreedySearch`.
        random_sampling_temp (int): See
            :class:`bioseq2seq.translate.greedy_search.GreedySearch`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`bioseq2seq.translate.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`bioseq2seq.translate.decode_strategy.DecodeStrategy`.
        tgt_prefix (bool): Force the predictions begin with provided -tgt.        
        replace_unk (bool): Replace unknown token.
        data_type (str): Source data type.
        verbose (bool): Print/log every translation.
        report_time (bool): Print/log total time/frequency.
        global_scorer (bioseq2seq.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
            self,
            model,
            fields,
            src_reader,
            tgt_reader,
            file_prefix,
            device="cpu",
            n_best=1,
            min_length=0,
            max_length=100,
            ratio=0.,
            beam_size=30,
            attn_save_layer=-1,
            random_sampling_topk=0,
            random_sampling_topp=0.0,
            random_sampling_temp=1,
            stepwise_penalty=None,
            dump_beam=False,
            block_ngram_repeat=0,
            ignore_when_blocking=frozenset(),
            replace_unk=False,
            ban_unk_token=False,
            tgt_prefix=False,
            phrase_table="",
            data_type="text",
            verbose=False,
            report_time=False,
            global_scorer = None,
            report_align=False,
            report_score=True,
            logger=None,
            seed=-1):

        self.model = model
        self.fields = fields
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_vocab_len = len(self._tgt_vocab)
        self._tgt_coding_idx = self._tgt_vocab.stoi["<PC>"]
        self._tgt_noncoding_idx = self._tgt_vocab.stoi["<NC>"]

        self._dev = device
        self._use_cuda = self._dev != "cpu"

        self.n_best = n_best
        self.max_length = max_length

        self.beam_size = beam_size
        self.attn_save_layer = attn_save_layer
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk
        self.sample_from_topp = random_sampling_topp
        
        self.min_length = min_length
        self.ban_unk_token = ban_unk_token
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {
            self._tgt_vocab.stoi[t] for t in self.ignore_when_blocking}
        self.src_reader = src_reader
        self.tgt_reader = tgt_reader
        self.replace_unk = replace_unk
        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError(
                "replace_unk requires an attentional decoder.")

        self.tgt_prefix = tgt_prefix
        self.phrase_table = phrase_table
        self.data_type = data_type
        self.verbose = verbose
        self.report_time = report_time

        self.global_scorer = global_scorer
        if self.global_scorer.has_cov_pen and \
                not self.model.decoder.attentional:
            raise ValueError(
                "Coverage penalty requires an attentional decoder.")

        self.pred_file = open(file_prefix+".preds",'w')
        self.self_attn_file = open(file_prefix+".self_attns",'w')
        self.enc_dec_attn_file = open(file_prefix+".enc_dec_attns",'w')
        self.failure_file = open(file_prefix+".failures",'w')

        self.report_align = report_align
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

        set_random_seed(seed, self._use_cuda)


    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _gold_score(self, batch, memory_bank, src_lengths,
                    enc_states, batch_size, src):
        
        if "tgt" in batch.__dict__:
            gs = self._score_target(
                batch, memory_bank, src_lengths)
            self.model.decoder.init_state(src, memory_bank, enc_states)
        else:
            gs = [0] * batch_size
        return gs

    def translate(
            self,
            src,
            names,
            cds,
            tgt=None,
            src_dir=None,
            batch_size=None,
            batch_type="sents",
            save_preds=False,
            save_SA=False,
            save_EDA=False,
            save_scores=False,
            align_debug=False,
            phrase_table=""):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            src: See :func:`self.src_reader.read()`.
            tgt: See :func:`self.tgt_reader.read()`.
            src_dir: See :func:`self.src_reader.read()` (only relevant
                for certain types of data).
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging
            align_debug (bool): enables the word alignment logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """

        if batch_size is None:
            raise ValueError("batch_size must be set")

        if self.tgt_prefix and tgt is None:
            raise ValueError("Prefix should be feed to tgt if -tgt_prefix.")

        
        src_data = {"reader": self.src_reader, "data": src, "dir": src_dir}
        tgt_data = {"reader": self.tgt_reader, "data": tgt, "dir": None}
        _readers, _data, _dir = inputters.Dataset.config(
            [('src', src_data), ('tgt', tgt_data)])

        data = inputters.Dataset(
            self.fields, readers=_readers, data=_data, dirs=_dir,
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred
        )

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=self._dev,
            batch_size=batch_size,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        xlation_builder = TranslationBuilder(data, self.fields, self.n_best,
                                             self.replace_unk, tgt, self.phrase_table)

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        #all_scores = []
        all_predictions = []
        #all_golds = []
        
        start_time = time.time()
        
        for batch in tqdm.tqdm(data_iter):
            #try:
            batch_data, src_cache = self.translate_batch(
                batch, data.src_vocabs, save_EDA
            )
            
            translations = xlation_builder.from_batch(batch_data)
            
            special = ['XR_949580.2', 'XR_001748355.1', 'XR_001707416.2', 'XR_003029405.1', 'XR_922291.3','XM_015134081.2', 'XM_032910311.1', 'NM_001375259.1', 'XR_002007359.1', 'XR_003726903.1']
            
            for b,idx in enumerate(batch.indices.data):
                tscript = names[idx]
                if tscript in special:
                    src, src_lens = batch.src
                    l = src_lens[b]
                    memory_bank = src_cache["memory_bank"][:l,b,:].cpu().numpy()
                    enc_states = src_cache["enc_states"][:l,b,:].cpu().numpy()
                    stored_src = src_cache["src"][:l,b,:].cpu().numpy()
                    filename = f'{tscript}_translator_3.npz'
                    np.savez(filename,src=stored_src,memory_bank=memory_bank,enc_states=enc_states)
            
            for b,trans in enumerate(translations):

                #all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                
                rna = "".join(trans.src_raw)
                transcript_name = names[trans.index]
                
                #print('transcript:',transcript_name,'\n')
                #print(trans.index,transcript_name)
                bounds = cds[trans.index]
                cds_bounds = None if bounds == "-1" else [int(x) for x in bounds.split(":")]
                
                if save_EDA:
                    # analyze encoder-decoder attention
                    enc_dec_attn = trans.context_attn
                    enc_dec_attn_state = EncoderDecoderAttentionDistribution(transcript_name,enc_dec_attn,\
                                                            rna,cds_bounds,attn_save_layer = self.attn_save_layer)
                    summary = enc_dec_attn_state.summarize()
                    self.enc_dec_attn_file.write(summary+"\n")
                if save_SA: 
                    # analyze self attention
                    self_attn = trans.self_attn
                    self_attn_state = SelfAttentionDistribution(transcript_name,self_attn,rna,cds_bounds)
                    summary = self_attn_state.summarize()
                    self.self_attn_file.write(summary+"\n")
                    self.self_attn_file.flush()
                    self.enc_dec_attn_file.flush()
                
                if tgt is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1
                    #all_golds.append("".join(trans.gold_sent))

                n_best_preds = ["".join(pred) for pred in trans.pred_sents[:self.n_best]]
                n_best_scores = [score for score in trans.pred_scores[:self.n_best]]
                #all_predictions += [n_best_preds]

                if save_preds:
                    self.pred_file.write("ID: {}\n".format(transcript_name))
                    self.pred_file.write("RNA: {}\n".format(rna))
                    for pred,score in zip(n_best_preds,n_best_scores):
                        self.pred_file.write("PRED: {} SCORE: {}\n".format(pred,score))
                    self.pred_file.write("PC_SCORE: {}\n".format(trans.coding_prob))
                    self.pred_file.write("GOLD: "+"".join(trans.gold_sent)+"\n\n")
                    self.pred_file.flush()
                
                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))
            ''' 
            except RuntimeError as err:
                print(err)
                torch.cuda.empty_cache()
                failed = names[batch.indices]
                print('GPU memory exceeded for transcript {}. Writing to failure file'.format(failed))
                self.failure_file.write("{}\n".format(failed))
                self.failure_file.flush()
            '''
        end_time = time.time()

        if self.report_score:
            msg = self._report_score('PRED', pred_score_total,
                                     pred_words_total)
            self._log(msg)
            if tgt is not None:
                msg = self._report_score('GOLD', gold_score_total,
                                         gold_words_total)
                self._log(msg)

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total translation time (s): %f" % total_time)
            #self._log("Average translation time (s): %f" % (
            #    total_time / len(all_predictions)))
            self._log("Tokens per second: %f" % (
                pred_words_total / total_time))

        if self.dump_beam:
            import json
            json.dump(self.translator.beam_accum,
                      codecs.open(self.dump_beam, 'w', 'utf-8'))
        
        self.self_attn_file.close()
        self.enc_dec_attn_file.close()
        self.pred_file.close()
        self.failure_file.close()

        #return all_predictions,all_golds,all_scores

    def _align_pad_prediction(self, predictions, bos, pad):
        """
        Padding predictions in batch and add BOS.
        Args:
            predictions (List[List[Tensor]]): `(batch, n_best,)`, for each src
                sequence contain n_best tgt predictions all of which ended with
                eos id.
            bos (int): bos index to be used.
            pad (int): pad index to be used.
        Return:
            batched_nbest_predict (torch.LongTensor): `(batch, n_best, tgt_l)`
        """
        dtype, device = predictions[0][0].dtype, predictions[0][0].device
        flatten_tgt = [best.tolist() for bests in predictions
                       for best in bests]
        paded_tgt = torch.tensor(
            list(zip_longest(*flatten_tgt, fillvalue=pad)),
            dtype=dtype, device=device).T
        bos_tensor = torch.full([paded_tgt.size(0), 1], bos,
                                dtype=dtype, device=device)
        full_tgt = torch.cat((bos_tensor, paded_tgt), dim=-1)
        batched_nbest_predict = full_tgt.view(
            len(predictions), -1, full_tgt.size(-1))  # (batch, n_best, tgt_l)
        return batched_nbest_predict

    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
         alignment src indice Tensor in size ``(batch, n_best,)``.
        """
        # (0) add BOS and padding to tgt prediction
        if hasattr(batch, 'tgt'):
            batch_tgt_idxs = batch.tgt.transpose(1, 2).transpose(0, 2)
        else:
            batch_tgt_idxs = self._align_pad_prediction(
                predictions, bos=self._tgt_bos_idx, pad=self._tgt_pad_idx)
        tgt_mask = (batch_tgt_idxs.eq(self._tgt_pad_idx) |
                    batch_tgt_idxs.eq(self._tgt_eos_idx) |
                    batch_tgt_idxs.eq(self._tgt_bos_idx))

        n_best = batch_tgt_idxs.size(1)
        # (1) Encoder forward.
        src, enc_states, memory_bank, src_lengths, dec_attn = self._run_encoder(batch)
        # (2) Repeat src objects `n_best` times.
        # We use batch_size x n_best, get ``(src_len, batch * n_best, nfeat)``
        src = tile(src, n_best, dim=1)
        enc_states = tile(enc_states, n_best, dim=1)
        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, n_best, dim=1) for x in memory_bank)
        else:
            memory_bank = tile(memory_bank, n_best, dim=1)
        src_lengths = tile(src_lengths, n_best)  # ``(batch * n_best,)``

        # (3) Init decoder with n_best src,
        self.model.decoder.init_state(src, memory_bank, enc_states)
        # reshape tgt to ``(len, batch * n_best, nfeat)``
        tgt = batch_tgt_idxs.view(-1, batch_tgt_idxs.size(-1)).T.unsqueeze(-1)
        dec_in = tgt[:-1]  # exclude last target from inputs
        _, attns = self.model.decoder(
            dec_in, memory_bank, memory_lengths=src_lengths, with_align=True)

        alignment_attn = attns["align"]  # ``(B, tgt_len-1, src_len)``
        # masked_select
        align_tgt_mask = tgt_mask.view(-1, tgt_mask.size(-1))
        prediction_mask = align_tgt_mask[:, 1:]  # exclude bos to match pred
        # get aligned src id for each prediction's valid tgt tokens
        alignement = extract_alignment(
            alignment_attn, prediction_mask, src_lengths, n_best)
        return alignement

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        with torch.no_grad():
            if self.sample_from_topk != 0 or self.sample_from_topp != 0:
                decode_strategy = GreedySearch(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    batch_size=batch.batch_size,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    return_attention=attn_debug or self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    keep_topp=self.sample_from_topp,
                    beam_size=self.beam_size,
                    ban_unk_token=self.ban_unk_token,
                )
            else:
                # TODO: support these blacklisted features
                assert not self.dump_beam
                decode_strategy = BeamSearch(
                    self.beam_size,
                    batch_size=batch.batch_size,
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length, max_length=self.max_length,
                    return_attention=attn_debug or self.replace_unk,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                    ban_unk_token=self.ban_unk_token,)

            return self._translate_batch_with_strategy(batch,decode_strategy)

    def _run_encoder(self, batch):
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                           else (batch.src, None)
    
        '''
        categories = np.asarray(list(range(18))).reshape(1,-1)
        enc = preprocessing.OneHotEncoder(categories=categories)
        raw = src.cpu().numpy().squeeze()
        one_hot_raw = enc.fit_transform(raw.reshape(-1,1)).toarray()
        '''

        enc_states, memory_bank, src_lengths, enc_attn = self.model.encoder(
            src, src_lengths)
       
        #idx = batch.indices.item()
        #plot_embeddings(enc_states,one_hot_raw,idx)

        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                               .type_as(memory_bank) \
                               .long() \
                               .fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths,enc_attn


    def _decode_and_generate(
            self,
            decoder_in,
            memory_bank,
            batch,
            memory_lengths,
            step=None):

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths,step=step,attn_save_layer=self.attn_save_layer
        )

        # Generator forward.
        if "std" in dec_attn:
            attn = dec_attn["std"]
        else:
            attn = None
        log_probs = self.model.generator(dec_out.squeeze(0))
        # returns [(batch_size x beam_size) , vocab ] when 1 step
        # or [ tgt_len, batch_size, vocab ] when full sentence

        return log_probs, attn

    def _translate_batch_with_strategy(
            self,
            batch,
            decode_strategy):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.
        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = batch.batch_size

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths, enc_attn = self._run_encoder(batch)
        src_cache = { "src" : src, "enc_states" : enc_states, "memory_bank" : memory_bank}
        
        self.model.decoder.init_state(src, memory_bank, enc_states)
        
        results = {
            "predictions": None,
            "scores": None,
            "self_attention": enc_attn,
            "context_attention" : None,
            "batch": batch,
            "gold_score": self._gold_score(
                batch, memory_bank, src_lengths,
                enc_states, batch_size, src)}
        
        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = None
        target_prefix = batch.tgt if self.tgt_prefix else None
                
        fn_map_state, memory_bank, memory_lengths, src_map = \
            decode_strategy.initialize(memory_bank, src_lengths, src_map,target_prefix=target_prefix)
        if fn_map_state is not None:
            self.model.decoder.map_state(fn_map_state)
       
        coding_probs = None

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                memory_lengths=memory_lengths,
                step=step)
            decode_strategy.advance(log_probs, attn)
            
            if step == 0:
                class_prob_indices = range(0,parallel_paths*batch_size,parallel_paths)
                coding_probs = [torch.exp(log_probs[i,self._tgt_coding_idx]).item() for i in class_prob_indices]
                noncoding_probs = [torch.exp(log_probs[i,self._tgt_noncoding_idx]).item() for i in class_prob_indices]
                #print(f'step ={step}, decoder_input = {decoder_input}')
                #print(f'prob(<PC>) = {coding_probs}')
                #print(f'prob(<NC>) = {noncoding_probs}') 

            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if parallel_paths > 1 or any_finished:
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["context_attention"] = decode_strategy.attention
        if results["self_attention"] is None:
            results["self_attention"] = [None for _ in range(batch_size)]
        results["coding_probs"] = coding_probs

        if self.report_align:
            results["alignment"] = self._align_forward(
                batch, decode_strategy.predictions)
        else:
            results["alignment"] = [[] for _ in range(batch_size)]
        return results, src_cache

    def _score_target(self, batch, memory_bank, src_lengths):
        tgt = batch.tgt
        tgt_in = tgt[:-1]
        log_probs, attn = self._decode_and_generate(
            tgt_in, memory_bank, batch,
            memory_lengths=src_lengths)

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[1:]
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores

    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name, score_total / words_total,
                name, math.exp(-score_total / words_total)))
        return msg

def plot_embeddings(embeddings,one_hot,idx):

    fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
    # make a little extra space between the subplots
    fig.subplots_adjust(hspace=0.5)

    # plot of embeddings
    data = embeddings.squeeze(1).transpose(0,1).cpu().numpy()
    ax1.imshow(data,aspect=10.0)
    # PSD of embeddings
    freq,ps = signal.welch(data,axis=1,scaling='density',average='median')
    n_dim, n_freq_bins = ps.shape
    for d in range(n_dim):
        ax2.plot(freq,ps[d,:],alpha=0.6)
  
    one_hot = one_hot.T
    # PSD of one-hot 
    freq,ps = signal.welch(one_hot,axis=1,scaling='density',average='median')
    n_dim, n_freq_bins = ps.shape
    for d in range(n_dim):
        ax3.plot(freq,ps[d,:],alpha=0.6)
    
    plt.savefig(f'transcript_{idx}.png')
    plt.close() 
