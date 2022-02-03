from scipy import stats
import orjson
import numpy as np
import sys
import matplotlib.pyplot as plt

def attribution(saved_file,transcript):
    with open(saved_file) as inFile:
        query = "{\"ID\": \""+transcript
        for l in inFile:
            if l.startswith(query):
                fields = orjson.loads(l)
                id_field = "ID"
                id = fields[id_field]
                #array = [float(x) for x in fields['ism_attr']] 
                array = [float(x) for x in fields['summed_attr']] 
                return np.asarray(array)
    return None

def calc_correlations(file_a, file_b):

    storage = []

    with open(file_a) as inFile:
        for l in inFile:
            fields = orjson.loads(l)
            tscript = fields['ID']
            is_pc = lambda x : x.startswith('NM_') or x.startswith('XM_')
            array = [float(x) for x in fields['summed_attr']] 
            scores_a = np.asarray(array)
            scores_b = attribution(file_b,tscript)
            if scores_b is not None:
                scores_b = scores_b[:scores_a.shape[0]]
                pearson_result = stats.pearsonr(scores_a,scores_b)
                kendall_result = stats.kendalltau(scores_a,scores_b)
                #storage.append(pearson_result[0])
                storage.append(kendall_result[0])

    
    dest = 'correlation_hist.svg'
    plt.hist(storage)
    plt.savefig(dest)
    plt.close()
    #print('Mean Pearson R = {}'.format(np.mean(storage)))
    print('Mean Kendall tau = {}, # samples = {}'.format(np.mean(storage),len(storage)))


if __name__ == "__main__":

    calc_correlations(sys.argv[1],sys.argv[2])
