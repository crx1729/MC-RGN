import numpy as np
import torch
import torch.nn.functional as F


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.act = torch.nn.ReLU()
        self.conv2 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.act(self.dropout1(self.conv1(inputs)))))
        outputs += inputs
        return outputs

class MC_RGN(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(MC_RGN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.MarkovWeight = args.MarkovWeight
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate, batch_first=True)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs, graph_emb, vision=False):
        seqs = graph_emb[torch.LongTensor(log_seqs).to(self.dev)]
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            if vision:
                mha_outputs, att_mat = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            else:
                mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)
        log_feats = self.last_layernorm(seqs)
        if vision:
            return log_feats, att_mat.squeeze(0).cpu().detach().numpy()
        else:
            return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, adj, McMat, indices):
        graph_emb = self.item_emb.weight + torch.sparse.mm(adj, self.item_emb.weight)
        log_feats = self.log2feats(log_seqs, graph_emb)
        pos_embs = graph_emb[torch.LongTensor(pos_seqs).to(self.dev)]
        neg_embs = graph_emb[torch.LongTensor(neg_seqs).to(self.dev)]
        McMat = McMat + self.last_layernorm.weight.mean()
        pos_logits = (log_feats * pos_embs).sum(dim=-1)[indices] + self.MarkovWeight * McMat[log_seqs[indices], pos_seqs[indices]]
        neg_logits = (log_feats * neg_embs).sum(dim=-1)[indices] + self.MarkovWeight * McMat[log_seqs[indices], neg_seqs[indices]]
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices, McMat, adj):
        graph_emb = self.item_emb.weight + torch.sparse.mm(adj, self.item_emb.weight)
        log_feats = self.log2feats(log_seqs, graph_emb, vision=False)
        final_feat = log_feats[:, -1, :]
        item_embs = graph_emb[torch.LongTensor(item_indices).to(self.dev)]
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits[0] + McMat[log_seqs.reshape(-1)[-1]][item_indices]