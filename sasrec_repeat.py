import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec_Repeat(torch.nn.Module):
    def __init__(self, user_num, item_num, repeat_num, args):
        super(SASRec_Repeat, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.repeat_num = repeat_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0) # +1いる？
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.repeat_emb = torch.nn.Embedding(self.repeat_num+1, args.hidden_units, padding_idx=0) # repeat embedding
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.concat_layer = torch.nn.Linear(
            args.hidden_units * (1 + 1), args.hidden_units
        )

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def repetitive_encoding(self, max_len, repeat, d_model):
        # print(f'repeat: {repeat[0]}')
        re = torch.zeros(self.batch_size, max_len, d_model).to(self.dev)
        # print(f're.shape: {re.shape}')
        rep = torch.LongTensor(repeat).to(self.dev).unsqueeze(-1)
        # print(f'rep.shape: {rep.shape}')
        # print(f'rep: {rep[0]}')
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).repeat(self.batch_size, 1).unsqueeze(1).to(self.dev)
        # print(f'div_term.shape: {div_term.shape}')
        re[:, :, 0::2] = torch.sin(rep * div_term)
        re[:, :, 1::2] = torch.cos(rep * div_term)
        # print(f're: {re[0]}')
        # print(f're.shape: {re.shape}')
        return re
    
    def positional_encoding(self, position, d_model):
        """
        Compute positional encoding as defined in the original Transformer paper.
        position: maximum sequence length.
        d_model: dimension of the model (embedding dimension).
        """
        pe = torch.zeros(position, d_model).to(self.dev)
        pos = torch.arange(0, position, dtype=torch.float).unsqueeze(1).to(self.dev)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).to(self.dev)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe
    
    def log2feats(self, log_seqs, log_repeat, rep_enc=False, pos_enc=False):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5 # これをrepeat mbeddingにも適用するかどうか、実験してみるしかないか
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        
        if rep_enc:
            max_length = log_seqs.shape[1]
            re = self.repetitive_encoding(max_length, log_repeat, self.item_emb.embedding_dim).to(self.dev)
            repeat = re
        else:
            repeat = self.repeat_emb(torch.LongTensor(log_repeat).to(self.dev)) # recboleでは.long()を使っている
        # repeat *= self.repeat_emb.embedding_dim ** 0.5 # repeatもスケーリング
        input_concat = torch.cat((seqs, repeat), -1)
        seqs = self.concat_layer(input_concat)
        
        if pos_enc:
            max_length = log_seqs.shape[1]
            pe = self.positional_encoding(max_length, self.item_emb.embedding_dim).to(self.dev)
            seqs += pe
        else:
            seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, log_repeat, pos_seqs, neg_seqs, rep_enc=False, pos_enc=False): # for training        
        log_feats = self.log2feats(log_seqs, log_repeat, rep_enc, pos_enc) # user_ids hasn't been used yet

        # BCE loss
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, log_repeat, item_indices): # for inference
        log_feats = self.log2feats(log_seqs, log_repeat) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
