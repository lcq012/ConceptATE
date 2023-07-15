import json
import xml.etree.ElementTree as ET
import json, random, nltk, re, unicodedata, os
import xml.etree.ElementTree as ET
from pdb import set_trace as stop
from transformers import BertTokenizer
from collections import Counter, OrderedDict
def repl_func(matched):
    a = matched.group(0)
    return ' ' + a.strip() + ' '

def get_token_and_label(sample):
    text = sample['text']
    text = unicodedata.normalize('NFKC', text)
    text_ = re.sub(r' [-*=/\']+(?![- ])', repl_func, text)
    text_ = re.sub(r'^[-*=/\']+(?![- ])', repl_func, text_)
    text_ = re.sub(r'(?<=[^- ])[-*=/\']+ ', repl_func, text_)
    text_ = re.sub(r'(?<=[^- ])[-*=/\']+$', repl_func, text_)
    text_ = re.sub(r'(?<=[^ ])[-/.*\'](?=[^ ])', repl_func, text_)
    tokens = nltk.tokenize.word_tokenize(text_)
    starts, ends = [0] * len(tokens), [0] * len(tokens)
    j, k = 0, 0
    fore_space = 0
    while text[fore_space] == " ":
        fore_space += 1
    for i, ch in enumerate(text[fore_space:]):
        if k == 0:
            if ch == ' ':
                continue
            starts[j] = i + fore_space
        if ch == '"':
            k += 1
        if k >= len(tokens[j]):
            ends[j] = i + fore_space
            k -= len(tokens[j])
            j += 1
            if ch == ' ':
                continue
            starts[j] = i + fore_space
        k += 1
    if ch != ' ':
        ends[j] = i + fore_space + 1
    
    labels = ["O"] * len(tokens)
    for aspect in sample["aspects"]:
        start = int(aspect["start"])
        end = int(aspect["end"])
        assert start in starts, f"wrong start, check the sentense:\n{text}"
        assert end in ends, f"wrong end, check the sentense:\n{text}"
        token_start = starts.index(start)
        token_end = ends.index(end) + 1
        labels[token_start] = "B-AS"
        for i in range(token_start + 1, token_end):
            labels[i] = "I-AS"
    return tokens, labels



# 在Tokenizer中加入特殊字符
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
special_tokens = ['[START]', '[END]']
tokenizer.add_tokens(special_tokens)


def processData(path:str): 
    # 解析xml文件
    tree = ET.parse(path)
    root = tree.getroot()
    context = {}
    # 遍历所有的句子，得到一句话的上下文
    for sentence in root.iter('sentence'):
        sentence_id = sentence.attrib['id']
        sample = {}
        sample["text"] = sentence.find("text").text
        aspectTerms = sentence.find("Opinions")
        sample["aspects"] = []
        if aspectTerms is not None:
            for term in aspectTerms.findall("Opinion"):
                asp = term.get("target")
                if asp == "NULL":
                    continue
                start = term.get("from")
                end = term.get("to")
                sample["aspects"].append({"term": asp, "start": start, "end": end})

        sentence_text = ' '.join(get_token_and_label(sample)[0])
        label_text = ' '.join(get_token_and_label(sample)[1])
        if sentence_id.split(':')[0] in context:
            context[sentence_id.split(':')[0]].append((sentence_id.split(':')[1], sentence_text, label_text))
        else:
            context[sentence_id.split(':')[0]] = [(sentence_id.split(':')[1], sentence_text, label_text)]

    # 将特殊符号加入的句子的前后两端
    senCon = {}
    for index, con in context.items():
        conn = sorted(con, key=lambda x: int(x[0]))
        for t in range(len(conn)):
            senCon[index+":"+conn[t][0]] = {}
            senCon[index+":"+conn[t][0]]['context'] = conn[t][1]
            senCon[index+":"+conn[t][0]]['label'] = conn[t][2]

    ## 处理成训练集与验证集，只适用于res15&16
    trains = []
    devs = []
    trainId = []
    devId = []
    # 解析xml文件
    for review in root.findall("Review"):
        for sentences in review.findall("sentences"):
            for sentence in sentences.findall("sentence"):
                if len(devs) < 150 and random.random() <= 0.3:
                    isTrain = False
                else:
                    isTrain = True
                # 列表中的第一项代表当前句子，第二项代表对应的标签，第三项代表上下文，第四项代表在上下文中当前句子对应的起始和终止位置
                item = []
                item.append(senCon[sentence.attrib['id']]['context'])
                item.append(senCon[sentence.attrib['id']]['label'])
                if not isTrain:
                    devId.append(sentence.attrib['id'].split(":")[0])
                    devs.append(item)
                else:
                    trainId.append(sentence.attrib['id'].split(":")[0])
                    trains.append(item)
    
    train_dict = Counter(trainId)
    dev_dict = Counter(devId)
    train_len = []
    dev_len = []
    for i in OrderedDict.fromkeys(trainId):
        train_len.append(train_dict[i])
    for i in OrderedDict.fromkeys(devId):
        dev_len.append(dev_dict[i])


    if "R15" in path:
        with open('/data/cqliu/ATEBaseline/contextStage2_seq/res15train.json', 'w') as f:
            json.dump(trains, f)
        with open('/data/cqliu/ATEBaseline/contextStage2_seq/res15dev.json', 'w') as f:
            json.dump(devs, f)
        with open('/data/cqliu/ATEBaseline/dataSeq/res15devseq.json', 'w') as f:
            json.dump(dev_len, f)
        with open('/data/cqliu/ATEBaseline/dataSeq/res15trainseq.json', 'w') as f:
            json.dump(train_len, f)

    elif "R16" in path:
        with open('/data/cqliu/ATEBaseline/contextStage2_seq/res16train.json', 'w') as f:
            json.dump(trains, f)
        with open('/data/cqliu/ATEBaseline/contextStage2_seq/res16dev.json', 'w') as f:
            json.dump(devs, f)
        with open('/data/cqliu/ATEBaseline/dataSeq/res16devseq.json', 'w') as f:
            json.dump(dev_len, f)
        with open('/data/cqliu/ATEBaseline/dataSeq/res16trainseq.json', 'w') as f:
            json.dump(train_len, f)

    print(str(len(devs) + len(trains)))



path16 = './R16/ABSA16_Restaurants_Train_SB1_v2.xml'
path15 = './R15/ABSA-15_Restaurants_Train_Final.xml'
processData(path16)