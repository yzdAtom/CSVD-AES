import copy
import time
import numpy as np
import pandas as pd
from model import get_net
import os
import ast
import logging
import warnings
warnings.filterwarnings("ignore")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import random
from query_strategies import WAAL
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from torch.utils.data import Dataset, DataLoader, SequentialSampler

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class FineTuneDataset(Dataset):
    def __init__(self, tokenizer, indexs_lb, file_path=None, block_size=None):
        self.examples = []
        index = 0
        with open(file_path, "r", encoding='ISO-8859-1') as f:
            for line in f.readlines():
                line = line.strip().split('<SPLIT>')
                example = convert_examples_to_features(line, tokenizer, block_size)
                if index not in indexs_lb:
                    self.examples.append(example)
                index += 1
        # random.shuffle(self.examples)
        cnt_1, cnt_0 = calculate(self.examples)
        self.cnt_1 = cnt_1
        self.cnt_0 = cnt_0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].input_metrics), torch.tensor(self.examples[i].label)




def evaluate(test_data_file, indexs_lb, device, fea, clf, tokenizer):
    block_size = min(512, tokenizer.max_len_single_sentence)
    eval_dataset = FineTuneDataset(tokenizer, indexs_lb, test_data_file, block_size)

    eval_batch_size = 32
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    logger.info("***** Running evaluation *****")
    logger.info("  Evaluation type = %s", type)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    nb_eval_steps = 0
    fea.eval()
    clf.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs_code_tid = batch[0].to(device)
        inputs_metrics = batch[1].to(device)
        label = batch[2].to(device)

        with torch.no_grad():
            latent = fea(inputs_code_tid, inputs_metrics)
            logit, context_embeddings = clf(latent)
            logits.append(logit[:,1].unsqueeze(1).cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1

    labels_1d = np.concatenate(labels).flatten()
    logits_1d = np.concatenate(logits).flatten()
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5
    eval_acc = np.mean(labels == preds)

    preds = np.zeros(labels_1d.shape[0])
    for index in range(labels_1d.shape[0]):
        if logits_1d[index] < 0.5:
            preds[index] = 0
        else:
            preds[index] = 1

    acc = accuracy_score(labels_1d, preds)
    pre = precision_score(labels_1d, preds)
    rec = recall_score(labels_1d, preds)
    f1 = f1_score(labels_1d, preds)
    auc = roc_auc_score(labels_1d, preds)
    mcc = matthews_corrcoef(labels_1d, preds)
    tn, fp, fn, tp = confusion_matrix(labels_1d, preds).ravel()
    if (fp + tn) == 0:
        fpr = -1.0
    else:
        fpr = float(fp) / (fp + tn)

    if (tp + fn) == 0:
        fnr = -1.0
    else:
        fnr = float(fn) / (tp + fn)

    logger.info("acc: %f", acc)
    logger.info("pre: %f", pre)
    logger.info("rec: %f", rec)
    logger.info("f1: %f", f1)
    logger.info("auc: %f", auc)
    logger.info("mcc: %f", mcc)
    logger.info("fpr: %f", fpr)
    logger.info("fnr: %f", fnr)
    result = {
        "eval_acc": round(eval_acc, 4),
        "eval_pre": round(pre, 4),
        "eval_rec": round(rec, 4),
        "eval_f1": round(f1, 4),
        "eval_auc": round(auc, 4),
        "eval_mcc": round(mcc, 4),
        "eval_fpr": round(fpr, 4),
        "eval_fnr": round(fnr, 4)
    }

    return result

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


def calculate(examles):
    cnt_0 = 0
    cnt_1 = 0
    for example in examles:
        if example.label == 0:
            cnt_0 += 1
        if example.label == 1:
            cnt_1 += 1

    return cnt_1, cnt_0

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

def transfer_all_dataset(tokenizer, file_path=None, block_size=None):  
    examples = []
    with open(file_path, "r", encoding='ISO-8859-1') as f:
        for line in f.readlines():
            line = line.strip().split('<SPLIT>')
            example = convert_examples_to_features(line, tokenizer, block_size)
            examples.append(example)
    cnt_1, cnt_0 = calculate(examples)
    all_cnt = cnt_0 + cnt_1
    return examples, all_cnt


def get_xy(datasets): 
    x_id = list()
    x_metrics = list()
    y = list()
    for dataset in datasets:
        code_token_id = dataset.input_ids
        code_metrics = dataset.input_metrics
        code_label = dataset.label
        x_id.append(code_token_id)
        x_metrics.append(code_metrics)
        y.append(code_label)
    x_id = np.array(x_id)
    x_metrics = np.array(x_metrics)
    y = torch.from_numpy(np.array(y))
    return x_id, x_metrics, y

def get_handler():
    return Wa_datahandler

def get_test_handler():
    return DataHandler

class Wa_datahandler(Dataset):
    def __init__(self,X_id_1, X_metrics_1, Y_1, X_id_2, X_metrics_2, Y_2):
        self.X_id_1 = X_id_1
        self.X_metrics_1 = X_metrics_1
        self.Y1 = Y_1
        self.X_id_2 = X_id_2
        self.X_metrics_2 = X_metrics_2
        self.Y2 = Y_2


    def __len__(self):
        return max(len(self.X_id_1),len(self.X_id_2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)
        # checking the index in the range or not
        if index < Len1:
            x_id_1 = self.X_id_1[index]
            x_metrics_1 = self.X_metrics_1[index]
            y_1 = self.Y1[index]
        else:
            # rescaling the index to the range of Len1
            re_index = index % Len1
            x_id_1 = self.X_id_1[re_index]
            x_metrics_1 = self.X_metrics_1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:
            x_id_2 = self.X_id_2[index]
            x_metrics_2 = self.X_metrics_2[index]
            y_2 = self.Y2[index]
        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2
            x_id_2 = self.X_id_2[re_index]
            x_metrics_2 = self.X_metrics_2[re_index]
            y_2 = self.Y2[re_index]
        return index,x_id_1,x_metrics_1,y_1,x_id_2,x_metrics_2,y_2


class DataHandler(Dataset):

    def __init__(self, X_id, X_metrics, Y):
        self.X_id = X_id
        self.X_metrics = X_metrics
        self.Y = Y

    def __getitem__(self, index):
        x_id, x_metrics, y = self.X_id[index], self.X_metrics[index], self.Y[index]
        return x_id, x_metrics, y, index

    def __len__(self):
        return len(self.X_id)






logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}


seed = 42
set_seed(seed)
source_projects = ['VLC']
target_projects = ['LibTIFF']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for source_project in source_projects:
    output_dir = './result/al_saved/%s' % source_project
    for target_project in target_projects:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO,
                            handlers=[
                                logging.FileHandler('./log/ALstage_%s_%s_%d.log' % (source_project, target_project, seed)),
                                logging.StreamHandler() 
                            ]
                            )
        for ratio in np.arange(0.3, 0.35, 0.1):  
            res = list()
            logger.info('================%s=>%s Select Ratio: %s start %d===============' % (source_project, target_project, ratio, seed))

            config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
            tokenizer = tokenizer_class.from_pretrained('microsoft/codebert-base')
            block_size = tokenizer.max_len_single_sentence
            block_size = min(512, tokenizer.max_len_single_sentence)
            source_data_file = '../cp_data/%s/%s_split/%s_train.txt' % (source_project, source_project, source_project)
            source_datasets, _ = transfer_all_dataset(tokenizer, source_data_file, block_size)
            x_label_id, x_label_metrics, y_label = get_xy(source_datasets)
            pool_data_file = '../cp_data/%s/%s_split/%s_train.txt' % (source_project, source_project, target_project)
            print(pool_data_file)
            test_data_file = '../cp_data/%s/%s_split/%s_test.txt' % (source_project, source_project, target_project)
            print(test_data_file)
            datasets, POOL_NUM = transfer_all_dataset(tokenizer, pool_data_file, block_size)
            x_pool_id, x_pool_metrics, y_pool = get_xy(datasets)
            print(POOL_NUM)
            SELECT_RATIO = ratio
            NUM_INIT_LB = 0  
            NUM_ROUND = 10
            NUM_QUERY = int(POOL_NUM * SELECT_RATIO / NUM_ROUND)  
            n_pool = len(y_pool)
            print('number of labeled pool: {}'.format(NUM_INIT_LB))
            print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
            epoch = 1
            idxs_lb = np.zeros(n_pool, dtype=bool) 
            idxs_tmp = np.arange(n_pool) 
            np.random.shuffle(idxs_tmp)  
            idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True  
            net_fea, net_clf = get_net()  
            train_handler = get_handler()
            test_handler = get_test_handler()
            strategy = WAAL(x_label_id, x_label_metrics, y_label, x_pool_id, x_pool_metrics, y_pool, idxs_lb, \
                            net_fea, net_clf, train_handler, test_handler)
            fea, clf = strategy.load_pretrain_dict()

            results = evaluate(test_data_file, [], device, fea, clf, tokenizer)
            for key, value in results.items():
                logger.info("  %s = %s", key, round(value, 4))
            column_list = list()
            column_list.append(results["eval_acc"])
            column_list.append(results["eval_pre"])
            column_list.append(results["eval_rec"])
            column_list.append(results["eval_f1"])
            column_list.append(results["eval_auc"])
            column_list.append(results["eval_mcc"])
            column_list.append(results["eval_fpr"])
            column_list.append(results["eval_fnr"])
            res.append(column_list)
            start_time = time.time()
            for rd in range(1, NUM_ROUND + 1):
                logger.info('================Round {:d}==============='.format(rd))

                q_idxs = strategy.query(NUM_QUERY)
                idxs_lb[q_idxs] = True
                # update
                strategy.update(idxs_lb)
                fea, clf = strategy.train(total_epoch=epoch)
            end_time = time.time()
            run_time = end_time - start_time
            indexs_lb = np.arange(n_pool)[idxs_lb] 
            results = evaluate(test_data_file, [], device, fea, clf, tokenizer)
            for key, value in results.items():
                logger.info("  %s = %s", key, round(value, 4))
            column_list = list()
            column_list.append(results["eval_acc"])
            column_list.append(results["eval_pre"])
            column_list.append(results["eval_rec"])
            column_list.append(results["eval_f1"])
            column_list.append(results["eval_auc"])
            column_list.append(results["eval_mcc"])
            column_list.append(results["eval_fpr"])
            column_list.append(results["eval_fnr"])
            res.append(column_list)
            idxs_prefix = 'idx_lb'
            idxs_output_dir = os.path.join(output_dir, '{}'.format(idxs_prefix))
            if not os.path.exists(idxs_output_dir):
                os.makedirs(idxs_output_dir)
            idxs_output_dir = os.path.join(idxs_output_dir, '{}'.format('%s_%s_%s_%d_idx_lb.npy' % (source_project, target_project, ratio, seed)))
            np.save(idxs_output_dir, idxs_lb)
            checkpoint_prefix = 'checkpoint-fea-clf'
            checkpoint_output_dir = os.path.join(output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(checkpoint_output_dir):
                os.makedirs(checkpoint_output_dir)
            fea_to_save = fea.module if hasattr(fea, 'module') else fea
            fea_output_dir = os.path.join(checkpoint_output_dir, '{}'.format('%s_%s_%s_%d_fea_last.bin' % (source_project, target_project, ratio, seed)))
            torch.save(fea_to_save.state_dict(), fea_output_dir)
            clf_to_save = clf.module if hasattr(clf, 'module') else clf
            clf_output_dir = os.path.join(checkpoint_output_dir, '{}'.format('%s_%s_%s_%d_clf_last.bin' % (source_project, target_project, ratio, seed)))
            torch.save(clf_to_save.state_dict(), clf_output_dir)
            logger.info("Saving model checkpoint")
            logger.info('================%s=>%s Select Ratio: %s end %d===============' % (source_project, target_project, ratio, seed))
            logger.info('\n\n\n\n\n')
            res_dataframe = pd.DataFrame(res, columns=["eval_acc", "eval_pre", "eval_rec", "eval_f1", "eval_auc", "eval_mcc", "eval_fpr", "eval_fnr"])
            res_dataframe.to_csv("./result/res_csv/%s/%s_%f_%d.csv" % (source_project, target_project, ratio, seed), index=False)
