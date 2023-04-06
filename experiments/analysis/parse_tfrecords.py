from tbparse import SummaryReader
import sys,os

def parse(directory,model,checkpoint_prefix,topk=5):

    all_files = sorted([os.path.join(directory,f) for f in os.listdir(directory)])
    cols = ['step', 'progress/accuracy', 'progress/class_accuracy', 'progress/lr',
                    'progress/patience','valid/accuracy', 'valid/class_accuracy','valid/patience']
    
    for i,f in enumerate(all_files):
        record = SummaryReader(f,pivot=True)
        data = record.scalars[cols]
        data = data.sort_values(by='valid/class_accuracy',ascending=False).iloc[:topk]
        acc = ','.join([f'{x:.2f}' for x in data['valid/class_accuracy'].tolist()]) 
        steps = data['step'].tolist()
        #print(f'# {model} replicate {i} valid/class_accuracy = {acc}') 
        for s in steps:
            checkpoint_name = f'{model}_{i+1}_{checkpoint_prefix}/_step_{s}.pt' 
            print(checkpoint_name) 

if __name__ == "__main__":

    parse(sys.argv[1],sys.argv[2],sys.argv[3])
