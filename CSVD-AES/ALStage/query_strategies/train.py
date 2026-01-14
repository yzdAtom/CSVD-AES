import os
import random
import ast
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler
from torch.autograd import grad
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
import torch.nn as nn
import torch.nn.init as init
MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

def initialize_classifier_weights(clf):
    for m in clf.modules():
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


class InputFeatures(object):
    def __init__(self,
                 input_tokens,
                 input_ids,
                 input_metrics,
                 label,
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.input_metrics = input_metrics
        self.label=label


def convert_examples_to_features(js,tokenizer, block_size):
    code=' '.join(js[1].split())
    code_tokens=tokenizer.tokenize(code)[:block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    metrics_data = ast.literal_eval(js[2])
    label = int(js[0])
    return InputFeatures(source_tokens, source_ids, metrics_data, label)


def gradient_penalty(critic, h_s, h_t):
    alpha = torch.rand(h_s.size(0), 1).cuda()
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty

def calculate(examles):
    cnt_0 = 0
    cnt_1 = 0
    for example in examles:
        if example.label == 0:
            cnt_0 += 1
        if example.label == 1:
            cnt_1 += 1
    print('*' * 100)
    print('True Positive samples:',cnt_1)
    print('True Negative samples:',cnt_0)
    print('*' * 100)
    return cnt_1, cnt_0


class FineTuneDataset(Dataset):
    def __init__(self, tokenizer, file_path=None, block_size=None):
        self.examples = []
        with open(file_path, "r", encoding='ISO-8859-1') as f:
            for line in f.readlines():
                line = line.strip().split('<SPLIT>')
                example = convert_examples_to_features(line, tokenizer, block_size)
                self.examples.append(example)
        cnt_1, cnt_0 = calculate(self.examples)
        self.cnt_1 = cnt_1
        self.cnt_0 = cnt_0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].input_metrics), torch.tensor(self.examples[i].label)



class WAAL:
    def __init__(self, Xl_id, Xl_metrics, Yl, Xp_id, Xp_metrics, Yp, idx_lb, \
                 net_fea, net_clf, train_handler, test_handler):

        self.Xl_id = Xl_id
        self.Xl_metrics = Xl_metrics
        self.Yl = Yl
        self.Xp_id = Xp_id
        self.Xp_metrics = Xp_metrics
        self.Yp = Yp
        self.idx_lb  = idx_lb
        self.net_fea = net_fea
        self.net_clf = net_clf
        self.train_handler = train_handler
        self.test_handler = test_handler
        self.n_pool = len(Yp)
        self.num_class = 2
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.selection = 10


    def update(self, idx_lb):
        self.idx_lb = idx_lb


    def load_pretrain_dict(self):
        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
        tokenizer = tokenizer_class.from_pretrained('microsoft/codebert-base')
        config = config_class.from_pretrained('microsoft/codebert-base')
        config.num_labels = 1
        model = model_class.from_pretrained('microsoft/codebert-base',
                                            from_tf=False,
                                            config=config,
                                            cache_dir=None,
                                            )

        self.fea = self.net_fea(model, config, tokenizer, 1).to(self.device)
        self.clf = self.net_clf().to(self.device)
        initialize_classifier_weights(self.clf)
        return self.fea, self.clf


    def train(self, total_epoch):
        print("[Training] labeled and unlabeled data")
        n_epoch = total_epoch

        self.fea.to(self.device)
        self.clf.to(self.device)

        lb_tensor = torch.tensor(self.idx_lb)
        zero_indices = torch.nonzero(self.Yp[lb_tensor] == 0).squeeze()
        num_ones = (self.Yp[lb_tensor] == 1).sum().item()
        if zero_indices.dim() == 0:
            zero_length = 0
        else:
            zero_length = len(zero_indices)
        if zero_length != 0:
            num_zeros_to_remove = zero_length - num_ones
            remove_indices = zero_indices[torch.randperm(len(zero_indices))[:num_zeros_to_remove]]
            run_idxs_lb = copy.deepcopy(self.idx_lb)
            true_indices = np.where(run_idxs_lb)[0]
            run_idxs_lb[true_indices[remove_indices.numpy()]] = False
        else:
            run_idxs_lb = copy.deepcopy(self.idx_lb)


        idx_lb_train = np.arange(self.n_pool)[run_idxs_lb]
        print(self.Yp[idx_lb_train])

        self.X_id_train = np.concatenate((self.Xl_id, self.Xp_id[idx_lb_train]), axis=0)
        self.X_metrics_train = np.concatenate((self.Xl_metrics, self.Xp_metrics[idx_lb_train]), axis=0)
        self.Y_train = torch.cat((self.Yl, self.Yp[idx_lb_train]), dim=0)
        cnt_1 = (self.Y_train == 1).sum().item()
        cnt_0 = (self.Y_train == 0).sum().item()
        loader_tr = DataLoader(
            self.test_handler(self.X_id_train, self.X_metrics_train, self.Y_train), shuffle=True, batch_size=16)

        max_steps = n_epoch * len(loader_tr)
        weight_decay = 0.01
        learning_rate = 2e-5
        adam_epsilon = 1e-8
        no_decay = ['bias', 'LayerNorm.weight']

        opt_fea_grouped_parameters = [
            {'params': [p for n, p in self.fea.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.fea.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        opt_fea = AdamW(opt_fea_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        fea_scheduler = get_linear_schedule_with_warmup(opt_fea, num_warmup_steps=max_steps * 0.1,
                                                    num_training_steps=max_steps)


        opt_clf_grouped_parameters = [
            {'params': [p for n, p in self.clf.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.clf.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        opt_clf = AdamW(opt_clf_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        clf_scheduler = get_linear_schedule_with_warmup(opt_clf, num_warmup_steps=max_steps * 0.1,
                                                    num_training_steps=max_steps)


        for epoch in range(n_epoch):
            self.fea.train()
            self.clf.train()
            Total_loss = 0
            n_batch = 0
            for label_x_id, label_x_metrics, label_y, idxs in loader_tr:
                n_batch += 1
                label_x_id, label_x_metrics, label_y = label_x_id.cuda(), label_x_metrics.cuda(), label_y.cuda()
                label_x_metrics = label_x_metrics.float()

                set_requires_grad(self.fea,requires_grad=True)
                set_requires_grad(self.clf,requires_grad=True)


                lb_z = self.fea(label_x_id, label_x_metrics)

                opt_fea.zero_grad()
                opt_clf.zero_grad()

                lb_out, _ = self.clf(lb_z)

                label_y = label_y.float()
                loss = torch.log(lb_out[:, 1] + 1e-10) * label_y * (cnt_0 / (cnt_0 + cnt_1)) + torch.log(
                    (1 - lb_out)[:, 1] + 1e-10) * (1 - label_y) * (cnt_1 / (cnt_0 + cnt_1))
                pred_loss = -loss.mean()
                print(pred_loss)


                loss = pred_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fea.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.clf.parameters(), 1.0)
                opt_fea.step()
                opt_clf.step()
                fea_scheduler.step()
                clf_scheduler.step()

                Total_loss += loss.item()

            Total_loss /= n_batch

            print('==========Inner epoch {:d} ========'.format(epoch))
            print('Training Loss {:.3f}'.format(Total_loss))

        return self.fea, self.clf


    def predict_prob(self,X_id, X_metrics, Y):

        loader_te = DataLoader(self.test_handler(X_id, X_metrics, Y), shuffle=False, batch_size=64)

        self.fea.eval()
        self.clf.eval()

        probs = torch.zeros([len(Y), self.num_class])
        with torch.no_grad():
            for x_id, x_metrics, y, idxs in loader_te:
                x_id, x_metrics, y = x_id.to(self.device), x_metrics.to(self.device), y.to(self.device)
                x_metrics = x_metrics.float()
                latent = self.fea(x_id, x_metrics)
                logit, context_embeddings = self.clf(latent)
                prob = torch.cat((logit[:,1].unsqueeze(1), (1 - logit)[:, 1].unsqueeze(1)), dim=1)
                probs[idxs] = prob.cpu()
        return probs


    def pred_dis_score(self,X_id, X_metrics, Y):
        loader_te = DataLoader(self.test_handler(X_id, X_metrics, Y), shuffle=False, batch_size=64)

        self.fea.eval()
        self.dis.eval()

        scores = torch.zeros(len(Y))

        with torch.no_grad():
            for x_id, x_metrics, y, idxs in loader_te:
                x_id, x_metrics, y = x_id.to(self.device), x_metrics.to(self.device), y.to(self.device)
                x_metrics = x_metrics.float()
                latent = self.fea(x_id, x_metrics)
                out = self.dis(latent).cpu()
                scores[idxs] = out.view(-1)
        return scores


    def single_worst(self, probas):

        value,_ = torch.max(-1*torch.log(probas),1)

        return value


    def L2_upper(self, probas):

        value = torch.norm(torch.log(probas),dim=1)

        return value


    def L1_upper(self, probas):

        value = torch.sum(-1*torch.log(probas),dim=1)

        return value


    def query(self,query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]
        probs = self.predict_prob(self.Xp_id[idxs_unlabeled], self.Xp_metrics[idxs_unlabeled], self.Yp[idxs_unlabeled])
        uncertainly_score = 0.5* self.L2_upper(probs) + 0.5* self.L1_upper(probs)
        total_score = uncertainly_score
        return idxs_unlabeled[total_score.sort()[1][:query_num]]

