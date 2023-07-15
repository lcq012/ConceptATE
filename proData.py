import xml.etree.ElementTree as ET
import json, random, nltk, re, unicodedata, os

def repl_func(matched):
    a = matched.group(0)
    return ' ' + a.strip() + ' '

def get_token_and_label(sample):
    text = sample['text']
    # text = "-Bluetooth (2.1), Fingerprint Reader, Full 1920x1080 screen *=Integrated Mic/Webcam* -Dual touchpad mode is interesting, and easy to use -5 USB ports -Runs about 38-41C on idle, Up to 65 (for me) on load -Very quiet -I could go on and on."
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
        
        # token_ch = tokens[j][k]
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
    cats = ["None"] * len(tokens)
    for aspect in sample["aspects"]:
        start = int(aspect["start"])
        end = int(aspect["end"])
        assert start in starts, f"wrong start, check the sentense:\n{text}"
        assert end in ends, f"wrong end, check the sentense:\n{text}"
        token_start = starts.index(start)
        token_end = ends.index(end) + 1
        labels[token_start] = "B-AS"
        cats[token_start] = aspect["cats"]
        for i in range(token_start + 1, token_end):
            labels[i] = "I-AS"
            cats[i] = aspect["cats"]
    return tokens, labels, cats


def write2file3(corpus, outpath):
    with open(outpath, "w", encoding="utf8") as f:
        for sample in corpus:
            data = list(zip(sample['tokens'], sample['labels'], sample["cats"]))
            line = '\n'.join(['\t'.join(w) for w in data])
            f.write(line + '\n\n')


## 处理成训练集与验证集，只适用于res15&16
def processNewTwoSet(path, output_path, devnum=150):
    tree = ET.parse(path)
    root = tree.getroot()
    trains = []
    devs = []
    for review in root.findall("Review"):
        for sentences in review.findall("sentences"):
            for sentence in sentences.findall("sentence"):
                if len(devs) < devnum and random.random() <= 0.3:
                    isTrain = False
                else:
                    isTrain = True
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
                        categ = " ".join(term.get("category").lower().split("#"))
                        sample["aspects"].append({"term": asp, "start": start, "end": end, "cats": categ})

                tokens, labels, cats = get_token_and_label(sample)
                sample["tokens"] = tokens
                sample["labels"] = labels
                # sample["cats"] = cats
                if not isTrain:
                    devs.append(sample)
                else:
                    trains.append(sample)
    write2file3(devs, os.path.join(output_path, "dev.txt"))
    write2file3(trains, os.path.join(output_path, "train.txt"))
    print(str(len(devs) + len(trains)))


## 处理成训练集与验证集，只适用于lap14&res14
def processOldTwoSet(path, output_path, devnum=150):
    tree = ET.parse(path)
    root = tree.getroot()
    trains = []
    devs = []
    for sentence in root:
        if len(devs) < devnum and random.random() <= 0.5:
            isTrain = False
        else:
            isTrain = True
        sample = {}
        sample["text"] = sentence.find("text").text
        aspects = sentence.find("aspectTerms")
        sample["aspects"] = []
        if aspects != None:
            aspects = aspects.findall("aspectTerm")
            for aspect in aspects:
                asp = aspect.attrib['term']
                start = aspect.attrib['from']
                end = aspect.attrib['to']
                sample["aspects"].append({"term": asp, "start": start, "end": end})
        tokens, labels = get_token_and_label(sample)
        sample["tokens"] = tokens
        sample["labels"] = labels
        if not isTrain:
            devs.append(sample)
        else:
            trains.append(sample)

    write2file3(devs, os.path.join(output_path, "dev.txt"))
    write2file3(trains, os.path.join(output_path, "train.txt"))
    print(str(len(devs) + len(trains)))