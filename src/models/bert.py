import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, AutoConfig
from vit_model import MLP


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    

class Bert(BertPreTrainedModel):
    def __init__(self, 
                 out_dim=768,
                 mlp_ratio=4.,
                 mlp_drop=0.,
                 config = AutoConfig.from_pretrained('bert-base-uncased')):
        super(Bert, self).__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)
        self.num_features = self.embed_dim = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlp = MLP(in_features=self.embed_dim, hidden_features=int(self.embed_dim * mlp_ratio), act_layer=nn.GELU , drop_ratio=mlp_drop, out_features=out_dim)
        self.post_init()
        self.apply(_init_vit_weights)

    def forward(self,x):
        # [batch_size, 3, seq_len]
        # 3 -> [input_ids, token_type_ids, attention_mask]
        print(x.shape)
        print("x:\n",x)
        # [batch_size, 3, seq_len]
        mapping = {'input_ids' : x[:,0,:], 'token_type_ids' : x[:,1,:], 'attention_mask' : x[:,2,:]}
        print(mapping)
        x = self.bert(**mapping)
        print(x.shape)
        # [batch_size, embed_dim]
        #x = x.last_hidden_state[:, 0, :]
        #print(x.shape)
        # [batch_size, out_dim]
        x = self.mlp(x)
        return x
    

        