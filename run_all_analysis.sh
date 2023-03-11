export CONFIG="PC.1.yaml"

# Table 1 (Classification) 
python bioseq2seq/bin/evaluate.py --config $CONFIG
python experiments/analysis/eval_tools.py experiments/competitor_results

# Fig 2 (mRNA and lncRNA translation) 
python experiments/analysis/eval_lncrna_translation.py --config $CONFIG align > noncoding_needle_cmds.txt
SGE_Array -c noncoding_needle_cmds.txt
python experiments/analysis/eval_mrna_translation.py --config $CONFIG align > coding_needle_cmds.txt
SGE_Array -c coding_needle_cmds.txt
python experiments/analysis/eval_mrna_translation.py --config $CONFIG parse 
python experiments/analysis/eval_lncrna_translation.py --config $CONFIG parse 

# Fig 3 (LFNet filters and attention PSD)
python experiments/analysis/plot_filters.py --config $CONFIG
python experiments/analysis/nucleotide_metagene.py --config $CONFIG 

# Fig 4 (ISM-agreement and self-agreement, example heatmaps)
python experiments/analysis/ensemble.py
python experiments/analysis/attr_correlations.py --config $CONFIG
python experiments/analysis/plot_examples.py --config $CONFIG 

# Fig 5 (MDIG mutation graphs, metagene, enriched motifs)
python experiments/analysis/mutation_analysis.py --config $CONFIG
python experiments/analysis/percentile_metagene.py --config $CONFIG 
python experiments/analysis/find_attr_spikes.py --config $CONFIG > streme_commands.txt
SGE_Array -c streme_commands.txt 
python experiments/analysis/parse_streme.py --config $CONFIG
