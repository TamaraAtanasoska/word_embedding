import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SkipGram(nn.Module):
    def __init__(self, configs, data_size, ng_dist=None, NGRAMS=False, n_max=None, pad_index=None):
        super(SkipGram, self).__init__()

        self.cfgs = configs
        self.vocab_size = data_size
        self.ng_dist = ng_dist
        self.embedding_dim = self.cfgs['EMBEDDING_DIM']
        self.ngrams = NGRAMS
        self.n_max = n_max

        self.in_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=pad_index)
        self.out_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=pad_index)

        torch.nn.init.uniform_(self.in_embeddings.weight, -1, 1)

        torch.nn.init.uniform_(self.out_embeddings.weight, -1, 1)
        with torch.no_grad():
            self.out_embeddings.weight[pad_index] = torch.zeros(self.embedding_dim)
            self.in_embeddings.weight[pad_index] = torch.zeros(self.embedding_dim)

    def forward_in(self, data):
        data = self.in_embeddings(data)
        data = data.reshape(-1, self.n_max, self.embedding_dim).sum(1) if self.ngrams else data
        return torch.nn.functional.normalize(data)

    def forward_out(self, data):
        return torch.nn.functional.normalize(self.out_embeddings(data))

    def forward_neg(self, updated_batch_size):
        ng_dist = torch.ones(self.vocab_size) if self.ng_dist is None else self.ng_dist
        ng_words = torch.multinomial(ng_dist,
                                     updated_batch_size * self.cfgs['NG_WORDS'],
                                     replacement=True
                                     ).to(device)

        # Reshape to size(BATCH_SIZE, NG_WORDS, EMBEDDING_DIM)
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
        in_embed = self.skipgram.forward_in(input_words).unsqueeze(2)

        out_embed = self.skipgram.forward_out(output_words).unsqueeze(1)

        updated_batch_size = in_embed.shape[0]
        ng_embed = self.skipgram.forward_neg(updated_batch_size).neg()
        ng_embed = torch.nan_to_num(ng_embed) if torch.isnan(ng_embed).any() or torch.isinf(ng_embed).any() else ng_embed

        out_loss = torch.bmm(out_embed, in_embed).sigmoid().log().squeeze()
        out_loss_check = torch.isnan(out_loss).any() or torch.isinf(out_loss).any()
        out_loss = torch.nan_to_num(out_loss, neginf=-torch.max(out_loss).item()-1, posinf=torch.max(out_loss).item()+1) if out_loss_check else out_loss

        neg_loss = torch.bmm(ng_embed, in_embed).sigmoid().log().squeeze().sum(1)
        neg_loss_check = torch.isnan(neg_loss).any() or torch.isinf(neg_loss).any()
        neg_loss = torch.nan_to_num(neg_loss, neginf=-torch.max(neg_loss).item()-1, posinf=torch.max(neg_loss).item()+1) if neg_loss_check else neg_loss
        neg_samp_loss = -(out_loss + neg_loss).mean()

        return neg_samp_loss
