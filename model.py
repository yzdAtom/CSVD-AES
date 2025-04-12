import numpy as np
import torch.nn as nn
import torch
import torch.nn.init as init

def get_net():
    return model_fea, model_clf

class model_fea(nn.Module):
    def __init__(self, encoder,config,tokenizer, head_num):
        super(model_fea, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.attention = nn.MultiheadAttention(embed_dim=768 * 2, num_heads=head_num)

    def forward(self, input_ids=None, code_metrics=None):
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1),
                                output_hidden_states=True, return_dict=True)
        cls_first = outputs.hidden_states[1][:,0,:]
        cls_last = outputs.hidden_states[-1][:,0,:]
        cls = cls_first + cls_last
        combined_features = torch.cat((cls, code_metrics), dim=1)
        combined_features = combined_features.unsqueeze(0)
        attention_output, _ = self.attention(combined_features, combined_features, combined_features)
        cls = attention_output.squeeze(0)
        return cls


class model_clf(nn.Module):
    def __init__(self):
        super(model_clf, self).__init__()
        self.fc = nn.Linear(768 * 2, 2)

    def forward(self, x):
        logits=self.fc(x)
        prob=torch.sigmoid(logits)
        prob = prob.reshape(prob.shape[0],-1)
        return prob, x



