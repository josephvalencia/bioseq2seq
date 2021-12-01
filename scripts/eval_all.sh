# Get validation metrics and tune ensemble
python bioseq2seq/bin/eval_top_beam.py --models_file bioseq2seq_nopenalty_val.txt --output_name bioseq2seq_val --tune_ensemble --top_n 3
python bioseq2seq/bin/eval_top_beam.py --models_file bioseq2seq_nopenalty_val.txt --output_name bioseq2seq_val --tune_ensemble --top_n 5

# get test metrics using top beam strategy
python bioseq2seq/bin/eval_top_beam.py --models_file bioseq2seq_nopenalty_test.txt --output_name bioseq2seq_test
python bioseq2seq/bin/eval_top_beam.py --models_file EDC_test.txt --output_name EDC_test
python bioseq2seq/bin/eval_top_beam.py --models_file bioseq2seq_no_penalty_best_ALL_tests.txt --output_name bioseq2seq_test_ALL
python bioseq2seq/bin/eval_top_beam.py --models_file EDC_no_penalty_best_ALL_tests.txt --output_name EDC_test_ALL

# get test metrics using beam ensemble strategy
python bioseq2seq/bin/eval_ensemble.py --ensemble_file bioseq2seq_val_top5.ensemble --output_name bioseq2seq_test 
python bioseq2seq/bin/eval_ensemble.py --ensemble_file bioseq2seq_no_penalty_best_ALL_tests.ensemble  --output_name bioseq2seq_test_ALL 

# get test metrics using beam ensemble strategy for consensus over multiple replicates
python bioseq2seq/bin/eval_ensemble_multi_replicate.py --ensemble_file bioseq2seq_val_top3.ensemble --output_name bioseq2seq_test 
