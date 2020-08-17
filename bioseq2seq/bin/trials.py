import subprocess,shlex,os

os.environ["PYTHONPATH"] = "/home/bb/valejose/home/bioseq2seq"
BIOHOME = "/home/bb/valejose/home/"

def train_one_trial(n_enc,n_dec,model_dim,pos,max_tokens,accum_steps):

    train_script = BIOHOME+"bioseq2seq/bioseq2seq/bin/train.py"
    input_file = BIOHOME+ "Fa/refseq_combined_cds.csv.gz"
    save_dir = BIOHOME +"bioseq2seq/checkpoints/coding_noncoding/"
    
    cmd_str = "python {} --input {} --num_gpus 4 --mode combined --save-directory {}\
    --n_enc_layers {} --n_dec_layers {} --model_dim {} --max_rel_pos {}\
    --max_tokens {} --accum_steps {}"

    cmd = cmd_str.format(train_script,input_file,save_dir,n_enc,n_dec,model_dim,pos,max_tokens,accum_steps)
    print("Running "+cmd)
    subprocess.call(shlex.split(cmd))

base = {'n_enc' : 4, 'n_dec' : 4, 'model_dim' : 128 , 'pos' : 10 , 'max_tokens' : 6000, 'accum_steps' : 1}
large = {'n_enc' : 6, 'n_dec' : 6 ,'model_dim' : 256, 'pos' : 10 , 'max_tokens' : 3000, 'accum_steps' : 4} 

settings = [large]

for s in settings:
    n_enc = s['n_enc']
    n_dec = s['n_dec']
    model_dim = s['model_dim']
    pos = s['pos']
    max_tokens = s['max_tokens']
    accum_steps = s['accum_steps']
    train_one_trial(n_enc,n_dec,model_dim,pos,max_tokens,accum_steps)

