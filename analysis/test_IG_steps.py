from scipy import stats
import orjson
import numpy as np

def attribution(saved_file,transcript):
    with open(saved_file) as inFile:
        query = "{\"ID\": \""+transcript
        for l in inFile:
            if l.startswith(query):
                fields = orjson.loads(l)
                id_field = "ID"
                id = fields[id_field]
                array = [float(x) for x in fields['summed_attr']] 
                return np.asarray(array)
    return None

#new_filename = 'seqseq_4_avg_pos_test_250steps.ig'
#old_filename = 'output/test/seq2seq/best_seq2seq_avg_pos_test.ig'

new_filename = "new_output/IG/seq2seq_3_T_pos_test.ig"
old_filename = "output/test/seq2seq/best_seq2seq_T_pos_test.ig" 

storage = []

with open(new_filename) as inFile:
    for l in inFile:
        fields = orjson.loads(l)
        tscript = fields['ID']
        is_pc = lambda x : x.startswith('NM_') or x.startswith('XM_')
        
        array = [float(x) for x in fields['summed_attr']] 
        higher_scores = np.asarray(array)
        lower_scores = attribution(old_filename,tscript) 
        lower_scores = lower_scores[:higher_scores.shape[0]]
        pearson_result = stats.pearsonr(higher_scores,lower_scores)
        kendall_result = stats.kendalltau(higher_scores,lower_scores)
        storage.append(pearson_result[0])
        print(pearson_result)
print('Mean Pearson R = {}'.format(np.mean(storage)))
