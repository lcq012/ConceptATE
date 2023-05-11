#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @author: leosan
# @time: 2021/12/19 13:34
# @contact: aszhongqiu@163.com
import sys
sys.path.append("..")
import numpy as np
import sklearn
import os
from tqdm import tqdm
from math import ceil
import shutil
import random
import time
import json
from logger import Logger, EvalLogger
import fitlog
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel,BertTokenizer,AlbertTokenizer,RobertaTokenizer,XLNetTokenizer,BertTokenizerFast
from transformers import TrainingArguments
from transformers import Trainer
from processing import load_dataset
from model.BERTED import myBert
from parameters import args
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score
from pdb import set_trace as stop
import codecs
fitlog.set_log_dir("../logs/")
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)  # 自动记录超参，超参必须作为全局变量

torch.set_printoptions(edgeitems=10)
myLogger = Logger(name='bert.log', save_file=args.log_file)
evalLogger = EvalLogger(args.performance_file)
logger = myLogger.get_logger()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
output_model_file = '../model/pytorch_model.bin'
#device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
print('device: ', device)

#mycode_start
def mycollect(batch):
    #print('\nppp',batch)
    batch_re={}
    c=[]
    p=[]
    uy=torch.tensor([1,3])
    a_uy=type(uy)
    f=[ i.shape[-1] if isinstance(i,a_uy) and i.shape !=torch.Size([]) else -1 for i in batch[0].values()]
    names = locals()
    for i in range(len(batch[0])):
        c.append(True)
        names['x%s' % i] = []
    for j in range(len(batch)):
        i=batch[j]
        d=[]
        count=0
        for a,b in i.items():
            if isinstance(b,a_uy) :
                if b.shape !=torch.Size([]):
                    d.append(b.shape[-1])
                else:
                    d.append(-1)
            else:
                d.append(-1)
            if j==0:
                p.append(a)
                names['x%s' % count]=[b]
            else:
                names['x%s' % count].append(b)
            count=count+1
        
        c=[ False if f[i]==-1 else  c[i] and f[i]==d[i] for i in range(len(f))]
        f=d
    for i in range(len(c)):
        if c[i]==True and p[i]!='com_input_ids' and p[i]!='com_attention_mask' and p[i]!='com_token_type_ids' and p[i]!='dep_tree':
            z=names['x%s' % i]
            batch_re[p[i]]=torch.cat([torch.unsqueeze(i,dim=0) for i in z],0)
        else:
            if p[i]=='act_lens':
                batch_re[p[i]]=torch.tensor([len(names['x%s' % i])])
            elif p[i]=='common_sense':
                batch_re[p[i]]=names['x%s' % i]
            elif p[i]=='sentence':
                batch_re[p[i]]=names['x%s' % i]
            elif p[i]=='com_input_ids':
                batch_re[p[i]]=names['x%s' % i]
            elif p[i]=='com_attention_mask':
                batch_re[p[i]]=names['x%s' % i]
            elif p[i]=='com_token_type_ids':
                batch_re[p[i]]=names['x%s' % i]
            elif p[i]=='dep_tree':
                batch_re[p[i]]=names['x%s' % i]
            elif p[i]=='pos_info':
                batch_re[p[i]]=names['x%s' % i]
            
    return batch_re

#mycode_end
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





### 评估模型
def eval_conlleval(args, tokens, total_gt, total_pt, convall_file, eval):
    # eg: 0:'O', 1:'B-AS', 2:'I-AS'
    id2label = {0:'O', 1:'B-ASP', 2:'I-ASP'}
    def test_result_to_pair(writer):
        for token, gt, pt in zip(tokens, total_gt, total_pt):
            assert len(token) == len(gt)
            assert len(token) == len(pt)
            line = ''
            for index in range(len(token)):
                cur_token = token[index]
                cur_gt = gt[index]
                cur_pt = pt[index]
                line += cur_token + ' ' + id2label[cur_gt] + ' ' + id2label[cur_pt] + '\n'
            writer.write(line + '\n')

    with codecs.open(convall_file, "w", encoding="utf-8") as writer:
        test_result_to_pair(writer)
    print('Here !!!!! ')
    from conlleval import return_report
    eval_result, p, r, f = return_report(convall_file)
    logger.info(''.join(eval_result))
    try:
        file_name = args.datasets + "-" + str(args.seed) + ".txt"
        with open(os.path.join(args.report_dir, file_name), "a+", encoding="utf8") as report:
            report.write(''.join(eval_result))
            report.write(eval + '\n')
            report.write("#" * 80 + "\n")
    except:
        raise
    return p, r, f       

def evaluate(args, model, tokenizer, eval_dataloader, eval):
    model.eval()
    total_tokens = list()
    total_gt_labels = list()
    total_pt_labels = list()
    total_merge_index = list()
    for step, batch in enumerate(eval_dataloader):
        inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "token_type_ids": batch["token_type_ids"].to(device),
                "labels": batch["labels"].to(device),
                "device":device,
                "com_input_ids": batch["com_input_ids"],
                "com_attention_mask": batch["com_attention_mask"],
                "com_token_type_ids": batch["com_token_type_ids"],
                "merge_index": batch["merge_index"]
            }
        tokenlist = [tokenizer.convert_ids_to_tokens(i, skip_special_tokens=True) for i in batch["input_ids"]]
        total_tokens.extend(tokenlist)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs[1]
            logits = logits.detach().cpu().numpy()
            # ground truth
            gt_labels = inputs["labels"].detach().cpu().numpy()
            merge_index = batch['merge_index'].detach().cpu().numpy()
            attention_mask = inputs["attention_mask"].detach().cpu().numpy()
            act_lens = batch["act_lens"].detach().cpu().numpy()

            # 原版是用attention mask取得，由于cls和sep也不需要计算进去，所以这么取的gt_labels >0的结点
            distill_boolean = gt_labels >= 0
            # ground truth
            final_gt = np.where(distill_boolean, gt_labels, np.nan)
            final_gt = [list(int(j) for j in i[~np.isnan(i)]) for i in final_gt] 
            total_gt_labels.extend(final_gt)
            # logits.shape: batch_size * max_seq_len * hidden_size
            predicts = np.argmax(logits, axis=-1)
            # predicted answers
            final_pt = np.where(distill_boolean, predicts, np.nan)
            final_pt = [list(int(j) for j in i[~np.isnan(i)]) for i in final_pt] 
            total_pt_labels.extend(final_pt)
            # process original labels
    print('total_bpe_words:', len(total_pt_labels))
    if args.save_result:
        with open('../log/logits_res', 'w', encoding='utf8') as fout:
            fout.write(str(total_pt_labels[:2000]) + '\n')
            fout.write(str(total_gt_labels[:2000]) + '\n')
            fout.write(str(total_merge_index[:2000]) + '\n')
    print('pt_labels', len(total_pt_labels))
    assert len(total_pt_labels) == len(total_gt_labels)

    precision, recall, f1 = eval_conlleval(args, total_tokens, total_gt_labels, total_pt_labels, os.path.join(args.output_dir, args.eval_test_file_name), eval)
    return precision, recall, f1






if args.seed > -1:
    seed_torch(args.seed)


def train(model):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None
    # print logger
    evalLogger.write('\n\n---- train stage ----\n')
    print('---- train stage ----')
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    evalLogger.write(localtime + '\n')

    best_res = tuple()

    # processing data
    tokenizer = select_tokenizer(args)

    train_dataset = load_dataset(args, tokenizer, "train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler = train_sampler, batch_size = args.train_batch_size,collate_fn=mycollect)

    dev_dataset = load_dataset(args, tokenizer, "dev")
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler = dev_sampler, batch_size = args.dev_batch_size,collate_fn=mycollect)

    test_dataset = load_dataset(args, tokenizer, "test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler = test_sampler, batch_size = args.test_batch_size,collate_fn=mycollect)


    batch_nums = len(train_dataloader)
    print("batch_nums", batch_nums)
    total_train_steps = batch_nums * args.epoch_nums

    logger.debug('total_train_steps:%d', total_train_steps)
    print('total_train_steps: ', total_train_steps)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup_steps = ceil(total_train_steps * args.warmup_rate)

    logger.debug('warmup_step:%d', warmup_steps)
    print('warm_up_steps: ', warmup_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps
    )

    # apex
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fptype)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # 读取断点 optimizer、scheduler
    if args.continue_checkpoint:
        checkpoint_dir = args.save_dir + "/checkpoint-" + str(args.checkpoint) + '-' + str(args.cur_batch)
        logger.debug('Load checkpoint:optimizer.pt & scheduler.pt from %s', checkpoint_dir)
        if os.path.isfile(os.path.join(checkpoint_dir, "optimizer.pt")):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))
        checkpoint = args.checkpoint + 1
    else:
          checkpoint = 0
    best_f1 = 0.0
    best_epoch_num = 0
    global_step = 0
    logging_loss = 0
    for epoch in range(args.epoch_nums):
        total_loss = 0
        model.train()
        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(localtime)
        print('epoch {} / {}'.format(epoch + 1, args.epoch_nums))
        logger.debug('epoch {} / {}'.format(epoch + 1, args.epoch_nums))
        evalLogger.write('epoch-' + str(epoch + 1) + '\n')
        for step, batch in tqdm(enumerate(train_dataloader),desc = "Training", total = batch_nums, ncols=50):
            model.zero_grad()
            optimizer.zero_grad()
            global_step += 1
            inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "token_type_ids": batch["token_type_ids"].to(device),
                    "labels": batch["labels"].to(device),
                    "device":device,
                    "com_input_ids": batch["com_input_ids"],
                    "com_attention_mask": batch["com_attention_mask"],
                    "com_token_type_ids": batch["com_token_type_ids"],
                    "merge_index": batch["merge_index"]
                }
            outputs = model(**inputs)
            loss = outputs['loss']
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            scheduler.step()
            ##############################################
            #####       My code Starting 
            if global_step % args.eval_steps == 0:
                fitlog.add_loss(loss.item(), name="Loss", step=global_step)
                model.eval()
                p, r, f = evaluate(args, model, tokenizer, dev_dataloader, eval="dev")
                fitlog.add_metric({"Dev": {"F1": f, "P": p, "R": r, "epoch": epoch}}, step=global_step)
                if epoch > 2 and f >= best_f1:
                    best_epoch_num = epoch
                    fitlog.add_best_metric({"Dev": {"F1": f, "P": p, "R": r, "epoch": epoch, "step": global_step}})
                    # 保存模型
                    torch.save(model.state_dict(), output_model_file)
                    if test_dataset is not None:
                        test_p, test_r, test_f = evaluate(args, model, tokenizer, test_dataloader, eval="test")
                        fitlog.add_metric({"Test": {"F1": test_f, "P": test_p, "R": test_r, "epoch": epoch}}, step=global_step)
                        fitlog.add_best_metric({"Test": {"F": test_f, "P": test_p, "R": test_r, "epoch": epoch, "step": global_step}})
                    best_f1 = f
                model.train()

    fitlog.add_best_metric({"Dev": {"F1": best_f1, "Epoch": best_epoch_num}})
    fitlog.finish()




def select_tokenizer(args):
    if "albert" in args.pretrain_model_name:
        return AlbertTokenizer.from_pretrained(args.pretrain_model_name)
    elif "roberta" in args.pretrain_model_name:
        return RobertaTokenizer.from_pretrained(args.pretrain_model_name)
    elif "bert" in args.pretrain_model_name:
        return BertTokenizerFast.from_pretrained(args.pretrain_model_name)
    elif "xlnet" in args.pretrain_model_name:
        return XLNetTokenizer.from_pretrained(args.pretrain_model_name)




def main():
    logger.info(args)
    # checkpoint_dir = args.save_dir + "/checkpoint-" + str(args.checkpoint) + '-' + str(args.cur_batch)
    checkpoint_dir = args.base_dir + "/checkpoint-" + str(args.checkpoint) + '-' + str(args.cur_batch)
    # print('check_point_dir',checkpoint_dir)
    model = myBert.from_pretrained( \
        args.pretrain_model_name, \
        num_labels=args.aspects_nums).to(device)
    # print(dir(model))
    if args.do_train:
        train(model)


if __name__ == '__main__':
    main()