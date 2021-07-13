import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SkipGram(nn.Module):
    def __init__(self, configs, data_size, ng_dist = None):
        super(SkipGram, self).__init__()

        self.cfgs = configs
        self.vocab_size = data_size
        self.ng_dist = ng_dist
        self.embedding_dim = self.cfgs['EMBEDDING_DIM']

        self.in_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.out_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        torch.nn.init.uniform_(self.in_embeddings.weight, -1, 1)
        torch.nn.init.uniform_(self.out_embeddings.weight, -1, 1)
   
    def forward_in(self, data):
        return self.in_embeddings(data)
    
    def forward_out(self, data):
        return self.out_embeddings(data)

    def forward_neg(self, updated_batch_size):
        ng_dist = torch.ones(self.vocab_size) if not self.ng_dist else self.ng_dist
        ng_words = torch.multinomial(ng_dist,
                                     updated_batch_size * self.cfgs['NG_WORDS'],
                                     replacement = True
                                    ).to(device)
        
        #Reshape to size(BATCH_SIZE, NG_WORDS, EMBEDDING_DIM) 
        ng_embeddings = self.out_embeddings(ng_words).view(updated_batch_size,
                                                          self.cfgs['NG_WORDS'],
                                                          self.cfgs['EMBEDDING_DIM'],
                                                         )
        return ng_embeddings


class NegativeSamplingLoss(nn.Module):
    def __init__(self, skipgram, configs):
        super(NegativeSamplingLoss, self).__init__()

        self.skipgram = skipgram
        self.cfgs = configs

    def forward(self, input_words, output_words):
        updated_batch_size = input_words.shape[0]

        in_embed = self.skipgram.forward_in(input_words).unsqueeze(2)
        out_embed = self.skipgram.forward_out(output_words).unsqueeze(1)
        ng_embed = self.skipgram.forward_neg(updated_batch_size).neg()
       
        out_loss = torch.bmm(in_embed, out_embed).squeeze().sigmoid().log()
        neg_loss = torch.bmm(ng_embed, in_embed).squeeze().sigmoid().log().sum(1)
        neg_samp_loss = -(out_loss.T + neg_loss).mean()

        return neg_samp_loss
