import torch
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

class PredictionWrapper(torch.nn.Module):
    
    def __init__(self,model,softmax):
        
        super(PredictionWrapper,self).__init__()
        self.model = model
        self.softmax = softmax 

    def forward(self,src,src_lens,decoder_input,batch_size):

        src = src.transpose(0,1)
        src, enc_states, memory_bank, src_lengths, enc_cache = self.run_encoder(src,src_lens,batch_size)

        self.model.decoder.init_state(src,memory_bank,enc_states)
        memory_lengths = src_lens
        
        scores, attn = self.decode_and_generate(
            decoder_input,
            memory_bank,
            memory_lengths=memory_lengths,
            step=0)

        return scores

    def run_encoder(self,src,src_lengths,batch_size):

        enc_states, memory_bank, src_lengths, enc_cache = self.model.encoder(src,src_lengths,grad_mode=False)

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
        
        scores = self.model.generator(dec_out.squeeze(0),softmax=self.softmax)
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
        self.class_token = self.tgt_vocab[tgt_class]
        self.sos_token = self.tgt_vocab['<s>']
        self.pc_token = self.tgt_vocab['<PC>']
        self.nc_token = self.tgt_vocab['<NC>']
        self.src_vocab = vocab["src"].base_field.vocab
        self.predictor = PredictionWrapper(self.model,self.softmax)

    def decoder_input(self,batch_size):
        return self.sos_token * torch.ones(size=(batch_size,1,1),dtype=torch.long).to(self.device)
   
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        self.run_fn(savefile,val_iterator,target_pos,baseline,transcript_names)
    
class EmbeddingAttribution(Attribution):
    '''Base class for attribution experiments using the dense embedding representation'''

    def __init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,batch_size=None):
        
        self.interpretable_emb = configure_interpretable_embedding_layer(self.model,'encoder.embeddings')
        super().__init__(model,device,vocab,tgt_class,softmax=softmax,sample_size=sample_size,batch_size=batch_size)

    def src_embed(self,src):
        src_emb = self.interpretable_emb.indices_to_embeddings(src.permute(1,0,2))
        src_emb = src_emb.permute(1,0,2)
        return src_emb 

