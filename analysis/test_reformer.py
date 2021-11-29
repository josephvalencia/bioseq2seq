import torch
from reformer_pytorch import Reformer, Recorder

'''
model = Reformer(
        dim = 512,
        depth = 6,
        heads = 8,
        lsh_dropout = 0.1,
        causal = True
        ).cuda()

model = Recorder(model)

x = torch.randn(2, 2048, 512).cuda()
y = model(x)

print(len(model.recordings[0]))
print(model.recordings[0][1]['attn'].shape) # a list of attention weights and buckets for the first forward pass

model.turn_off() # stop recording
model.turn_on() # start recording
model.clear() # clear the recordings

model = model.eject() # recover the original model and remove all listeners
'''

import torch
from reformer_pytorch import ReformerLM, Recorder

DE_SEQ_LEN = 2048
EN_SEQ_LEN = 512

encoder = ReformerLM(
    num_tokens = 4,
    emb_dim = 128,
    dim = 128,
    depth = 6,
    heads = 8,
    max_seq_len = DE_SEQ_LEN,
    fixed_position_emb = True,
    return_embeddings = True # return output of last attention layer
    ).cuda()

encoder = Recorder(encoder)

decoder = ReformerLM(
    num_tokens = 29,
    emb_dim = 128,
    dim = 128,
    depth = 6,
    heads = 8,
    max_seq_len = EN_SEQ_LEN,
    fixed_position_emb = True,
    causal = True
    ).cuda()

decoder = Recorder(decoder)

x  = torch.randint(0, 4, (1, DE_SEQ_LEN)).long().cuda()
yi = torch.randint(0, 29, (1, EN_SEQ_LEN)).long().cuda()

enc_keys = encoder(x)               # (1, 4096, 1024)
print(enc_keys.shape)
yo = decoder(yi, keys = enc_keys)   # (1, 4096, 20000)

print(len(encoder.recordings[0]))
print(encoder.recordings[0][1]['attn'].shape) # a list of attention weights and buckets for the first forward pass

print(len(decoder.recordings[0]))
print(decoder.recordings[0][1]['attn'].shape) # a list of attention weights and buckets for the first forward pass
