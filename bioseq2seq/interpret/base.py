import torch
from functools import partial,reduce
from typing import Union
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from approxISM.embedding import TensorToOneHot, OneHotToEmbedding 

class PredictionWrapper(torch.nn.Module):
    
    def __init__(self,model,softmax):
        
        super(PredictionWrapper,self).__init__()
        self.model = model
        self.softmax = softmax 

    def forward(self,src,src_lens,decoder_inputs,batch_size):
        
        src = src.transpose(0,1)
        src, enc_states, memory_bank, src_lengths, enc_cache = self.run_encoder(src,src_lens,batch_size)

        self.model.decoder.init_state(src,memory_bank,enc_states)
        memory_lengths = src_lens
       
        for i,dec_input in enumerate(decoder_inputs):
            scores, attn = self.decode_and_generate(
                dec_input,
                memory_bank,
                memory_lengths=memory_lengths,
                step=i)
            #print(i,torch.argmax(torch.nn.functional.softmax(scores)))
        return scores

    def run_encoder(self,src,src_lengths,batch_size):

        enc_states, memory_bank, src_lengths, enc_cache = self.model.encoder(src,src_lengths,grad_mode=False)
        #print(f'enc_states = {enc_states.shape}, memory_bank = {memory_bank.shape}, src_lengths = {src_lengths.shape}')
        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank).long().fill_(memory_bank.size(0))

        return src, enc_states, memory_bank, src_lengths,enc_cache

    def decode_and_generate(self,decoder_in, memory_bank, memory_lengths, step=None):

        dec_out, dec_attn = self.model.decoder(decoder_in,
                                            memory_bank,
                                            memory_lengths=memory_lengths,
                                            step=step,grad_mode=True)

        if "std" in dec_attn:
            attn = dec_attn["std"]
        else:
            attn = None
        #print(f'dec_out before squeeze = {dec_out.shape}')    
        scores = self.model.generator(dec_out.squeeze(0),softmax=self.softmax)
        #scores = self.model.generator(dec_out,softmax=self.softmax)
        return scores, attn

class Attribution:
    '''Base class for attribution experiments '''
    def __init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,batch_size=None):
        
        self.device = device
        self.model = model
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.softmax = softmax
        self.tgt_vocab = vocab['tgt'].base_field.vocab
        self.class_token = self.tgt_vocab[tgt_class] if tgt_class != 'GT' else 'GT'
        self.sos_token = self.tgt_vocab['<s>']
        self.pc_token = self.tgt_vocab['<PC>']
        self.nc_token = self.tgt_vocab['<NC>']
        self.src_vocab = vocab["src"].base_field.vocab
        self.predictor = PredictionWrapper(self.model,self.softmax)

    def decoder_input(self,batch_size,prefix=None):
        
        if prefix is None:
            ones =  torch.ones(size=(batch_size,1,1),dtype=torch.long).to(self.device)
            return (self.sos_token*ones,)
        else:
            chunked = [x.repeat(batch_size,1,1) for x in torch.tensor_split(prefix,prefix.shape[0],dim=0)]
            return tuple(chunked)
            
    def get_raw_src(self,src):
        saved_src = src.detach().cpu().numpy().ravel()
        #return ''.join([self.src_vocab.itos[c] for c in saved_src])
        return [self.src_vocab.itos[c] for c in saved_src]

    def get_src_token(self,char):
        return self.src_vocab.stoi[char]
    
    def input_grads(self,outputs,inputs):
        total_pred = outputs.sum()
        total_pred.backward(torch.ones_like(total_pred),inputs=inputs)
        grads = inputs.grad
        #grads -= grads.mean(dim=-1,keepdims=True)
        return grads
            
    def class_likelihood_ratio(self,pred_classes,class_token):
        counterfactual = [x for x in range(pred_classes.shape[1]) if x != class_token]
        counter_idx = torch.tensor(counterfactual,device=pred_classes.device)
        counterfactual_score = pred_classes.index_select(2,counter_idx)
        return pred_classes[:,:,class_token] - counterfactual_score.sum()
    
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        self.run_fn(savefile,val_iterator,target_pos,baseline,transcript_names)
    
class EmbeddingAttribution(Attribution):
    '''Base class for attribution experiments using the dense embedding representation'''

    def __init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,batch_size=None):
        
        self.interpretable_emb = configure_interpretable_embedding_layer(model,'encoder.embeddings')
        super().__init__(model,device,vocab,tgt_class,softmax=softmax,sample_size=sample_size,batch_size=batch_size)

    def src_embed(self,src):
        src_emb = self.interpretable_emb.indices_to_embeddings(src.permute(1,0,2))
        src_emb = src_emb.permute(1,0,2)
        return src_emb 

    def nucleotide_embed(self,src,nucleotide):

        src_size = list(src.size())
        # retrieve embedding from torchtext
        n = self.src_vocab[nucleotide]
        # copy across length
        test =  n*torch.ones_like(src).to(self.device)
        baseline_emb = self.interpretable_emb.indices_to_embeddings(test.permute(1,0,2))
        baseline_emb = baseline_emb.permute(1,0,2)
        return baseline_emb

def get_module_by_name(parent: Union[torch.Tensor, torch.nn.Module],
                               access_string: str):
    names = access_string.split(sep='.')
    return reduce(getattr, names, parent)

class OneHotGradientAttribution(Attribution):

    def __init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,batch_size=None,times_input=False,smoothgrad=False):
        
        self.smoothgrad = smoothgrad
        # augment Embedding with one hot utilities
        embedding_modulelist = get_module_by_name(model,'encoder.embeddings.make_embedding.emb_luts')
        old_embedding = embedding_modulelist[0] 
        self.onehot_embed_layer = TensorToOneHot(old_embedding)
        dense_embed_layer = OneHotToEmbedding(old_embedding)
        embedding_modulelist[0] = dense_embed_layer
        self.predictor = PredictionWrapper(model,softmax)
        
        super().__init__(model,device,vocab,tgt_class,softmax=softmax,sample_size=sample_size,batch_size=batch_size)
    
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        raise NotImplementedError
