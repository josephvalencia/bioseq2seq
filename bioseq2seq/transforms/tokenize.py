"""Transforms relate to tokenization/subword."""
from bioseq2seq.utils.logging import logger
from bioseq2seq.transforms import register_transform
from .transform import Transform, ObservableStats


class TokenizerTransform(Transform):
    """Tokenizer transform abstract class."""

    def __init__(self, opts):
        """Initialize necessary options for Tokenizer."""
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Available options relate to Subword."""
        # Sharing options among `TokenizerTransform`s, same name conflict in
        # this scope will be resolved by remove previous occurrence in parser
        group = parser.add_argument_group(
            'Transform/Subword/Common', conflict_handler='resolve',
            description=".. Attention:: Common options shared by all subword transforms. "  # noqa: E501
            "Including options for indicate subword model path, "
            "`Subword Regularization <https://arxiv.org/abs/1804.10959>`_"
            "/`BPE-Dropout <https://arxiv.org/abs/1910.13267>`_, "
            "and `Vocabulary Restriction <https://github.com/rsennrich/subword-nmt#best-practice-advice-for-byte-pair-encoding-in-nmt>`__.")  # noqa: E501
        group.add('-src_subword_model', '--src_subword_model',
                  help="Path of subword model for src (or shared).")
        group.add("-tgt_subword_model", "--tgt_subword_model",
                  help="Path of subword model for tgt.")

        # subword regularization(or BPE dropout) options:
        group.add('-src_subword_nbest', '--src_subword_nbest',
                  type=int, default=1,
                  help="Number of candidates in subword regularization. "
                       "Valid for unigram sampling, "
                       "invalid for BPE-dropout. "
                       "(source side)")
        group.add('-tgt_subword_nbest', '--tgt_subword_nbest',
                  type=int, default=1,
                  help="Number of candidates in subword regularization. "
                       "Valid for unigram sampling, "
                       "invalid for BPE-dropout. "
                       "(target side)")
        group.add('-src_subword_alpha', '--src_subword_alpha',
                  type=float, default=0,
                  help="Smoothing parameter for sentencepiece unigram "
                       "sampling, and dropout probability for BPE-dropout. "
                       "(source side)")
        group.add('-tgt_subword_alpha', '--tgt_subword_alpha',
                  type=float, default=0,
                  help="Smoothing parameter for sentencepiece unigram "
                       "sampling, and dropout probability for BPE-dropout. "
                       "(target side)")

        # subword vocabulary restriction options:
        group.add('-src_subword_vocab', '--src_subword_vocab',
                  type=str, default="",
                  help="Path to the vocabulary file for src subword. "
                  "Format: <word>\t<count> per line.")
        group.add("-tgt_subword_vocab", "--tgt_subword_vocab",
                  type=str, default="",
                  help="Path to the vocabulary file for tgt subword. "
                  "Format: <word>\t<count> per line.")
        group.add('-src_vocab_threshold', '--src_vocab_threshold',
                  type=int, default=0,
                  help="Only produce src subword in src_subword_vocab with "
                  " frequency >= src_vocab_threshold.")
        group.add("-tgt_vocab_threshold", "--tgt_vocab_threshold",
                  type=int, default=0,
                  help="Only produce tgt subword in tgt_subword_vocab with "
                  " frequency >= tgt_vocab_threshold.")

    @classmethod
    def _validate_options(cls, opts):
        """Extra checks for Subword options."""
        assert 0 <= opts.src_subword_alpha <= 1, \
            "src_subword_alpha should be in the range [0, 1]"
        assert 0 <= opts.tgt_subword_alpha <= 1, \
            "tgt_subword_alpha should be in the range [0, 1]"

    def _parse_opts(self):
        self.share_vocab = self.opts.share_vocab
        self.src_subword_model = self.opts.src_subword_model
        self.tgt_subword_model = self.opts.tgt_subword_model
        self.src_subword_nbest = self.opts.src_subword_nbest
        self.tgt_subword_nbest = self.opts.tgt_subword_nbest
        self.src_subword_alpha = self.opts.src_subword_alpha
        self.tgt_subword_alpha = self.opts.tgt_subword_alpha
        self.src_subword_vocab = self.opts.src_subword_vocab
        self.tgt_subword_vocab = self.opts.tgt_subword_vocab
        self.src_vocab_threshold = self.opts.src_vocab_threshold
        self.tgt_vocab_threshold = self.opts.tgt_vocab_threshold

    def _repr_args(self):
        """Return str represent key arguments for TokenizerTransform."""
        kwargs = {
            'share_vocab': self.share_vocab,
            'src_subword_model': self.src_subword_model,
            'tgt_subword_model': self.tgt_subword_model,
            'src_subword_alpha': self.src_subword_alpha,
            'tgt_subword_alpha': self.tgt_subword_alpha,
            'src_subword_vocab': self.src_subword_vocab,
            'tgt_subword_vocab': self.tgt_subword_vocab,
            'src_vocab_threshold': self.src_vocab_threshold,
            'tgt_vocab_threshold': self.tgt_vocab_threshold
        }
        return ', '.join([f'{kw}={arg}' for kw, arg in kwargs.items()])


class SubwordStats(ObservableStats):
    """Runing statistics for counting tokens before/after subword transform."""

    __slots__ = ["subwords", "words"]

    def __init__(self, subwords: int, words: int):
        self.subwords = subwords
        self.words = words

    def update(self, other: "SubwordStats"):
        self.subwords += other.subwords
        self.words += other.words

    def __str__(self) -> str:
        return "{}: {} -> {} tokens".format(
            self.name(), self.words, self.subwords
        )


@register_transform(name='sentencepiece')
class SentencePieceTransform(TokenizerTransform):
    """SentencePiece subword transform class."""

    def __init__(self, opts):
        """Initialize necessary options for sentencepiece."""
        super().__init__(opts)

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        import sentencepiece as spm
        spm.set_random_generator_seed(seed)

    def warm_up(self, vocabs=None):
        """Load subword models."""
        super().warm_up(None)
        import sentencepiece as spm
        load_src_model = spm.SentencePieceProcessor()
        load_src_model.Load(self.src_subword_model)
        _diff_vocab = self.src_subword_vocab != self.tgt_subword_vocab or \
            self.src_vocab_threshold != self.tgt_vocab_threshold
        if self.src_subword_vocab != "" and self.src_vocab_threshold > 0:
            load_src_model.LoadVocabulary(
                self.src_subword_vocab, self.src_vocab_threshold)
        if self.share_vocab and not _diff_vocab:
            self.load_models = {
                'src': load_src_model,
                'tgt': load_src_model
            }
        else:
            load_tgt_model = spm.SentencePieceProcessor()
            load_tgt_model.Load(self.tgt_subword_model)
            if self.tgt_subword_vocab != "" and self.tgt_vocab_threshold > 0:
                load_tgt_model.LoadVocabulary(
                    self.tgt_subword_vocab, self.tgt_vocab_threshold)
            self.load_models = {
                'src': load_src_model,
                'tgt': load_tgt_model
            }

    def _tokenize(self, tokens, side='src', is_train=False):
        """Do sentencepiece subword tokenize."""
        sp_model = self.load_models[side]
        sentence = ' '.join(tokens)
        nbest_size = self.tgt_subword_nbest if side == 'tgt' else \
            self.src_subword_nbest
        if is_train is False or nbest_size in [0, 1]:
            # derterministic subwording
            segmented = sp_model.encode(sentence, out_type=str)
        else:
            # subword sampling when nbest_size > 1 or -1
            # alpha should be 0.0 < alpha < 1.0
            alpha = self.tgt_subword_alpha if side == 'tgt' else \
                self.src_subword_alpha
            segmented = sp_model.encode(
                sentence, out_type=str, enable_sampling=True,
                alpha=alpha, nbest_size=nbest_size)
        return segmented

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply sentencepiece subword encode to src & tgt."""
        src_out = self._tokenize(example['src'], 'src', is_train)
        tgt_out = self._tokenize(example['tgt'], 'tgt', is_train)
        if stats is not None:
            n_words = len(example['src']) + len(example['tgt'])
            n_subwords = len(src_out) + len(tgt_out)
            stats.update(SubwordStats(n_subwords, n_words))
        example['src'], example['tgt'] = src_out, tgt_out
        return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        kwargs_str = super()._repr_args()
        additional_str = 'src_subword_nbest={}, tgt_subword_nbest={}'.format(
            self.src_subword_nbest, self.tgt_subword_nbest
        )
        return kwargs_str + ', ' + additional_str


@register_transform(name='bpe')
class BPETransform(TokenizerTransform):
    """subword_nmt: official BPE subword transform class."""

    def __init__(self, opts):
        """Initialize necessary options for subword_nmt."""
        super().__init__(opts)

    def _parse_opts(self):
        super()._parse_opts()
        self.dropout = {'src': self.src_subword_alpha,
                        'tgt': self.tgt_subword_alpha}

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        import random
        random.seed(seed)

    def warm_up(self, vocabs=None):
        """Load subword models."""
        super().warm_up(None)
        from subword_nmt.apply_bpe import BPE, read_vocabulary
        # Load vocabulary file if provided and set threshold
        src_vocabulary, tgt_vocabulary = None, None
        if self.src_subword_vocab != "" and self.src_vocab_threshold > 0:
            with open(self.src_subword_vocab, encoding='utf-8') as _sv:
                src_vocabulary = read_vocabulary(_sv, self.src_vocab_threshold)
        if self.tgt_subword_vocab != "" and self.tgt_vocab_threshold > 0:
            with open(self.tgt_subword_vocab, encoding='utf-8') as _tv:
                tgt_vocabulary = read_vocabulary(_tv, self.tgt_vocab_threshold)
        # Load Subword Model
        with open(self.src_subword_model, encoding='utf-8') as src_codes:
            load_src_model = BPE(codes=src_codes, vocab=src_vocabulary)
        if self.share_vocab and (src_vocabulary == tgt_vocabulary):
            self.load_models = {
                'src': load_src_model,
                'tgt': load_src_model
            }
        else:
            with open(self.tgt_subword_model, encoding='utf-8') as tgt_codes:
                load_tgt_model = BPE(codes=tgt_codes, vocab=tgt_vocabulary)
            self.load_models = {
                'src': load_src_model,
                'tgt': load_tgt_model
            }

    def _tokenize(self, tokens, side='src', is_train=False):
        """Do bpe subword tokenize."""
        bpe_model = self.load_models[side]
        dropout = self.dropout[side] if is_train else 0.0
        segmented = bpe_model.segment_tokens(tokens, dropout=dropout)
        return segmented

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply bpe subword encode to src & tgt."""
        src_out = self._tokenize(example['src'], 'src', is_train)
        tgt_out = self._tokenize(example['tgt'], 'tgt', is_train)
        if stats is not None:
            n_words = len(example['src']) + len(example['tgt'])
            n_subwords = len(src_out) + len(tgt_out)
            stats.update(SubwordStats(n_subwords, n_words))
        example['src'], example['tgt'] = src_out, tgt_out
        return example


@register_transform(name='bioseq2seq_tokenize')
class ONMTTokenizerTransform(TokenizerTransform):
    """OpenNMT Tokenizer transform class."""

    def __init__(self, opts):
        """Initialize necessary options for OpenNMT Tokenizer."""
        super().__init__(opts)

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        import pybioseq2seqtok
        pybioseq2seqtok.set_random_seed(seed)

    @classmethod
    def add_options(cls, parser):
        """Available options relate to Subword."""
        super().add_options(parser)
        group = parser.add_argument_group('Transform/Subword/ONMTTOK')
        group.add('-src_subword_type', '--src_subword_type',
                  type=str, default='none',
                  choices=['none', 'sentencepiece', 'bpe'],
                  help="Type of subword model for src (or shared) "
                       "in pybioseq2seqtok.")
        group.add('-tgt_subword_type', '--tgt_subword_type',
                  type=str, default='none',
                  choices=['none', 'sentencepiece', 'bpe'],
                  help="Type of subword model for tgt in  pybioseq2seqtok.")
        group.add('-src_bioseq2seqtok_kwargs', '--src_bioseq2seqtok_kwargs', type=str,
                  default="{'mode': 'none'}",
                  help="Other pybioseq2seqtok options for src in dict string, "
                  "except subword related options listed earlier.")
        group.add('-tgt_bioseq2seqtok_kwargs', '--tgt_bioseq2seqtok_kwargs', type=str,
                  default="{'mode': 'none'}",
                  help="Other pybioseq2seqtok options for tgt in dict string, "
                  "except subword related options listed earlier.")

    @classmethod
    def _validate_options(cls, opts):
        """Extra checks for OpenNMT Tokenizer options."""
        super()._validate_options(opts)
        src_kwargs_dict = eval(opts.src_bioseq2seqtok_kwargs)
        tgt_kwargs_dict = eval(opts.tgt_bioseq2seqtok_kwargs)
        if not isinstance(src_kwargs_dict, dict):
            raise ValueError("-src_bioseq2seqtok_kwargs isn't a dict valid string.")
        if not isinstance(tgt_kwargs_dict, dict):
            raise ValueError("-tgt_bioseq2seqtok_kwargs isn't a dict valid string.")
        opts.src_bioseq2seqtok_kwargs = src_kwargs_dict
        opts.tgt_bioseq2seqtok_kwargs = tgt_kwargs_dict

    def _parse_opts(self):
        super()._parse_opts()
        self.src_subword_type = self.opts.src_subword_type
        self.tgt_subword_type = self.opts.tgt_subword_type
        logger.info("Parsed pybioseq2seqtok kwargs for src: {}".format(
            self.opts.src_bioseq2seqtok_kwargs))
        logger.info("Parsed pybioseq2seqtok kwargs for tgt: {}".format(
            self.opts.tgt_bioseq2seqtok_kwargs))
        self.src_other_kwargs = self.opts.src_bioseq2seqtok_kwargs
        self.tgt_other_kwargs = self.opts.tgt_bioseq2seqtok_kwargs

    @classmethod
    def get_specials(cls, opts):
        src_specials, tgt_specials = set(), set()
        if opts.src_bioseq2seqtok_kwargs.get("case_markup", False):
            _case_specials = ['｟mrk_case_modifier_C｠',
                              '｟mrk_begin_case_region_U｠',
                              '｟mrk_end_case_region_U｠']
            src_specials.update(_case_specials)
        if opts.tgt_bioseq2seqtok_kwargs.get("case_markup", False):
            _case_specials = ['｟mrk_case_modifier_C｠',
                              '｟mrk_begin_case_region_U｠',
                              '｟mrk_end_case_region_U｠']
            tgt_specials.update(_case_specials)
        return (set(), set())

    def _get_subword_kwargs(self, side='src'):
        """Return a dict containing kwargs relate to `side` subwords."""
        subword_type = self.tgt_subword_type if side == 'tgt' \
            else self.src_subword_type
        subword_model = self.tgt_subword_model if side == 'tgt' \
            else self.src_subword_model
        subword_nbest = self.tgt_subword_nbest if side == 'tgt' \
            else self.src_subword_nbest
        subword_alpha = self.tgt_subword_alpha if side == 'tgt' \
            else self.src_subword_alpha
        kwopts = dict()
        if subword_type == 'bpe':
            kwopts['bpe_model_path'] = subword_model
            kwopts['bpe_dropout'] = subword_alpha
        elif subword_type == 'sentencepiece':
            kwopts['sp_model_path'] = subword_model
            kwopts['sp_nbest_size'] = subword_nbest
            kwopts['sp_alpha'] = subword_alpha
        else:
            logger.warning('No subword method will be applied.')
        vocabulary_threshold = self.tgt_vocab_threshold if side == 'tgt' \
            else self.src_vocab_threshold
        vocabulary_path = self.tgt_subword_vocab if side == 'tgt' \
            else self.src_subword_vocab
        if vocabulary_threshold > 0 and vocabulary_path != "":
            kwopts['vocabulary_path'] = vocabulary_path
            kwopts['vocabulary_threshold'] = vocabulary_threshold
        return kwopts

    def warm_up(self, vocabs=None):
        """Initialize Tokenizer models."""
        super().warm_up(None)
        import pybioseq2seqtok
        src_subword_kwargs = self._get_subword_kwargs(side='src')
        src_tokenizer = pybioseq2seqtok.Tokenizer(
            **src_subword_kwargs, **self.src_other_kwargs
        )
        tgt_subword_kwargs = self._get_subword_kwargs(side='tgt')
        _diff_vocab = (
            src_subword_kwargs.get('vocabulary_path', '') !=
            tgt_subword_kwargs.get('vocabulary_path', '') or
            src_subword_kwargs.get('vocabulary_threshold', 0) !=
            tgt_subword_kwargs.get('vocabulary_threshold', 0))
        if self.share_vocab and not _diff_vocab:
            self.load_models = {
                'src': src_tokenizer,
                'tgt': src_tokenizer
            }
        else:
            tgt_subword_kwargs = self._get_subword_kwargs(side='tgt')
            tgt_tokenizer = pybioseq2seqtok.Tokenizer(
                **tgt_subword_kwargs, **self.tgt_other_kwargs
            )
            self.load_models = {
                'src': src_tokenizer,
                'tgt': tgt_tokenizer
            }

    def _tokenize(self, tokens, side='src', is_train=False):
        """Do OpenNMT Tokenizer's tokenize."""
        tokenizer = self.load_models[side]
        sentence = ' '.join(tokens)
        segmented, _ = tokenizer.tokenize(sentence)
        return segmented

    def _detokenize(self, tokens, side='src', is_train=False):
        """Do OpenNMT Tokenizer's detokenize."""
        tokenizer = self.load_models[side]
        detokenized = tokenizer.detokenize(tokens)
        return detokenized

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply OpenNMT Tokenizer to src & tgt."""
        src_out = self._tokenize(example['src'], 'src')
        tgt_out = self._tokenize(example['tgt'], 'tgt')
        if stats is not None:
            n_words = len(example['src']) + len(example['tgt'])
            n_subwords = len(src_out) + len(tgt_out)
            stats.update(SubwordStats(n_subwords, n_words))
        example['src'], example['tgt'] = src_out, tgt_out
        return example

    def apply_reverse(self, translated):
        """Apply OpenNMT Tokenizer to src & tgt."""
        return self._detokenize(translated.split(), 'tgt')

    def _repr_args(self):
        """Return str represent key arguments for class."""
        repr_str = '{}={}'.format('share_vocab', self.share_vocab)
        repr_str += ', src_subword_kwargs={}'.format(
            self._get_subword_kwargs(side='src'))
        repr_str += ', src_bioseq2seqtok_kwargs={}'.format(self.src_other_kwargs)
        repr_str += ', tgt_subword_kwargs={}'.format(
            self._get_subword_kwargs(side='tgt'))
        repr_str += ', tgt_bioseq2seqtok_kwargs={}'.format(self.tgt_other_kwargs)
        return repr_str
