export CONFIG = Dec3.ini

# Fig 2 (EDC vs bioseq2seq)
python analysis/boxplot_compare.py --config $CONFIG

# Table 1 and 2
python b

# Fig 3 (self-attn)
python analysis/self_attn_analysis.py --config $CONFIG

# Fig 4 (power spectrum)
python analysis/average_attentions.py --config $CONFIG

# Fig 5 (enriched motifs)
python analysis/attribution_analysis.py --config $CONFIG
python analysis/run_streme.py --config $CONFIG
python analysis/parse_streme.py --config $CONFIG

# Fig 6 (MDIG)
python analysis/mutation_analysis.py --config $CONFIG


