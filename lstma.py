class LinearMultiHeadAttention(nn.Module):
  
    def __init__(self, d_model: int, n_head: int, dropout: float,
                 layer_idx=None, feature_map=None, eps=1e-6):
        super(LinearMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.eps = eps
        self.layer_idx = layer_idx
        assert d_model % n_head == 0, "d_model should be divisible by n_head"
        self.head_dim = d_model // n_head

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.feature_map = (
            feature_map(d_model) if feature_map else
            elu_feature_map(d_model)
        )

    def reset_params(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, query, key, value, key_mask=None, state=None):
        q = self.q_proj(query)
        k = v = None

        # if self.layer_idx is not None:
        #     if isinstance(state, LinearTransformerState):
        #         k = state.encoder_key[self.layer_idx]
        #         v = state.encoder_value[self.layer_idx]
        #
        if k is None:
            k = self.k_proj(key)
            v = self.v_proj(value)
        #
        # if self.layer_idx is not None:
        #     if isinstance(state, LinearTransformerState):
        #         state.encoder_key[self.layer_idx] = k
        #         state.encoder_value[self.layer_idx] = v

        batch_size, q_len, d_model = q.size()
        k_len, v_len = k.size(1), v.size(1)

        q = q.reshape(batch_size, q_len, self.n_head, self.head_dim)
        k = k.reshape(batch_size, k_len, self.n_head, self.head_dim)
        v = v.reshape(batch_size, v_len, self.n_head, self.head_dim)

        self.feature_map.new_feature_map()
        q = self.feature_map.forward_queries(q)
        k = self.feature_map.forward_keys(k)

        if key_mask is not None:
            _key_mask = ~key_mask[:, :, None, None].bool()
            k = k.masked_fill(_key_mask, 0.0)

        # KV = torch.einsum("bsnd,bsnm->bnmd", K, V)
        #
        # Z = 1 / (torch.einsum("bsnd,bnd->bsn", Q, K.sum(dim=1))+self.eps)
        #
        # V_new = torch.einsum("bsnd,bnmd,bsn->bsnm", Q, KV, Z)
        V_new = linear_attention(q, k, v, self.eps)

        output = V_new.contiguous().reshape(batch_size, q_len, -1)
        output = self.o_proj(output)

        return output
      
class TransformerEncoderLayer3(nn.Module):

    def __init__(self,d_model,nhead,seq_length,dim_feedforward=2048,dropout=0.1):
        super(TransformerEncoderLayer3, self).__init__()
        self.d_model=d_model
        self.self_attn = LinearMultiHeadAttention(d_model=d_model, n_head=nhead, dropout=dropout)
        self.self_attn2 = LinearMultiHeadAttention(d_model=seq_length,n_head=10, dropout=dropout

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout =nn.Dropout(dropout)
        self.linear2 =nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)
        # self.dropout3=nn.Dropout(p=0.8)
        self.activation =nn.ReLU()

        # self.fc=nn.Linear(self.d_model,2)

    def forward(self, src):
        # src2 = self.self_attn(src,src,src)
        # src2=src.permute(0, 2, 1)
        src3=src
        src3=src3.permute(0,2,1)
        src2 = self.self_attn2(src3, src3, src3)
        src2 = src2.permute(0, 2, 1)
        src2 = self.self_attn(src2,src2,src2)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
 
class Self_encoder(nn.Module):
    def __init__(self,d_model,nhead,num_layers,seq_length,pos_dropout,fc_dropout=0.1):
        super(Self_encoder, self).__init__()
        self.d_model=d_model
#       self.pos_enc=PositionalEncoding1(self.d_model,pos_dropout,seq_length)
        self.encoder_layer3 = TransformerEncoderLayer3(d_model=d_model, nhead=nhead, seq_length=seq_length)
        self.fc=nn.Linear(self.d_model+4,2)
        self.dropout=nn.Dropout(fc_dropout)

    def forward(self,x,Pheno_Info):
        # x=self.pos_enc(x)
        # print(x.shape)
        # x2=x
        x=self.encoder_layer3(x)
        # x = F.avg_pool1d(x, kernel_size=2)
        # x2=self.encoder_layer2(x2)
        # print("encoder===",type(x))
        x=torch.sum(x,dim=1)
        # x2=torch.sum(x2, dim=2)
        # print("x.shape_Time", x.shape)
        # print("x2.shape_Space", x2.shape)
        # x=x.view(x.size(0),-1)

        x = torch.cat((x, Pheno_Info), dim=1)

        x=self.fc(x)
        # print("fc======",x.shape)
        # x=self.dropout(x)
        return x                                                  
