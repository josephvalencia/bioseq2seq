import tqdm
import torch
import bioseq2seq.inputters as inputters

from bioseq2seq.translate import Translator
from bioseq2seq.translate.translation import Translation, TranslationBuilder

class TransparentTranslationBuilder(TranslationBuilder):

    def from_batch(self, translation_batch):

        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) == len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, self_attn, context_attn, align, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["self_attention"],
                        translation_batch["context_attention"],
                        translation_batch["alignment"],
                        translation_batch["gold_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        if not any(align):  # when align is a empty nested list
            align = [None] * batch_size

        # Sorting
        inds, perm = torch.sort(batch.indices)
        if self._has_text_src:
            src = batch.src[0][:, :, 0].index_select(1, perm)
        else:
            src = None
        tgt = batch.tgt[:, :, 0].index_select(1, perm) \
            if self.has_tgt else None

        translations = []

        for b in range(batch_size):

            info = (preds[b][0],tgt[1:,b])
            translations.append(info)

        return translations

class TransparentTranslator(Translator):
    ''''Modified Translator class that exposes raw integer output and attentions'''

    def translate(
            self,
            src,
            names,
            cds,
            tgt=None,
            attribution = None,
            src_dir=None,
            batch_size=None,
            batch_type="sents",
            save_preds=False,
            save_attn=False,
            align_debug=False,
            phrase_table=""):
        
        if batch_size is None:
            raise ValueError("batch_size must be set")

        src_data = {"reader": self.src_reader, "data": src, "dir": src_dir}
        tgt_data = {"reader": self.tgt_reader, "data": tgt, "dir": None}

        _readers, _data, _dir = inputters.Dataset.config([('src', src_data), ('tgt', tgt_data)])

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

        xlation_builder = TransparentTranslationBuilder(data,
                                                        self.fields, 
                                                        self.n_best,
                                                        self.replace_unk, 
                                                        tgt, 
                                                        self.phrase_table)
        
        all_translations = []
        
        for batch in tqdm.tqdm(data_iter):

            batch_data = self.translate_batch(
                batch, data.src_vocabs, save_attn
            )

            translations = xlation_builder.from_batch(batch_data)
            
            for trans in translations:

                self_attn = trans.enc_attn
                enc_dec_attn = trans.context_attn
                gold = trans.enc_attn()

                # gradients = attribution.attribute()

