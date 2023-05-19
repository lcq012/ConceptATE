import json
import csv
import nltk
import stanza
# nltk.download('averaged_perceptron_tagger')
rel_all_A=["CapableOf","Causes","IsA","MannerOf","MotivatedByGoal","ReceivesAction"]
rel_all_B=["capable of","causes","is a","manner of","motivated by goal","receives action"]
pos_select=['FW','JJ','JJR','JJS','NN','NNS','NNP','NNPS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ']
pos_al=['RB', ',', 'PRP', 'MD', 'VB', 'IN', 'DT', 'NN', '.', 'NNS', 'VBP', 'RP', 'PRP$', 'VBD', 'VBG', 'JJ', 'NNP', 'CD', 'VBN', 'VBZ', 'WRB', 'TO', 'WP', 'CC', ':', 'EX', 'NNPS', 'PDT', 'WDT', 'JJS', 'RBR', 'POS', '$', 'JJR', 'RBS', 'UH', 'FW', 'WP$', "''", 'SYM', '``', '#']
# nltk.download('tagsets')
# nltk.help.brown_tagset()
def read_example(file_name):
    #,download_method=None
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    with open(file_name,'r',encoding="utf8") as fin:
        _corpus = json.load(fin)
        for traget in _corpus:
            context, label_str = traget[0], traget[1]
            print('\n',context)

            labels = label_str.split(" ")
            
            assert len(context.split(" ")) == len(labels)
            u=context.split(" ")
            temp_ex=u
            len_example=len(temp_ex)
            if len_example>128:
                len_example=128
            doc = nlp(context)
            list_src=[ word.head-1  for sent in doc.sentences for word in sent.words if word.head!=0 and word.head<=128 and word.id<=128]+list(range(len_example))
            list_dst=[ word.id-1 for sent in doc.sentences for word in sent.words if word.head!=0 and word.head<=128 and word.id<=128]+list(range(len_example))
            list_src=[str(i) for i in list_src]
            list_dst=[str(i) for i in list_dst]
            pos_example=[n for m,n in nltk.pos_tag(u)]
            temp=[]
            tre=[]
            pos_add=" ".join(pos_example)
            tree_add=[]
            tree_add.append(" ".join(list_src))
            tree_add.append(" ".join(list_dst))
            for i in range(len(u)):
                if pos_example[i] in pos_select:
                    i_num,y_num = search(u[i],'./entity2id.txt','./need.csv',2)
                else:
                    i_num=[u[i]]
                    y_num=['-1']
                #print('0000',type(i_num),i_num,[i])
                s = " " .join (i_num)
                t = " " .join (y_num)
                temp.append(s)
                tre.append(t)
            traget.append(pos_add)
            traget.append(tree_add)
            traget.append(temp)
            traget.append(tre)
        with open('dev01.json', 'w') as f:
            json.dump(_corpus, f ,indent=4)


def search(word,entity,csv_com,step):
    rel_entity=[]
    rel_num=[]
    #relation_entity=['-1']
    rel_entity.append(word)
    rel_num.append('-1')
    if word=='people':
        return rel_entity,rel_num
    #with open(csv_com) as csv_conceptnet:
    csv_conceptnet=open(csv_com)
    conceptnet_reader=csv.reader(csv_conceptnet, delimiter=',')
    l=0
    for row in conceptnet_reader:
        l+=1
        if l==1:continue
        relation,start,end=row[0],row[1],row[2]
        if start==word and end not in rel_entity: 
            rel_entity.append(end)
            rel_num.append(str(rel_all_A.index(relation)))
    csv_conceptnet.close()
    return rel_entity,rel_num

# def search2(word,entity,csv_com,step):
#     rel_entity=[]
#     relation_entity=['-1']
#     rel_entity.append(word)
#     csv_conceptnet=open(csv_com)
#     conceptnet_reader=csv.reader(csv_conceptnet, delimiter=',')
#     l=0
#     for row in conceptnet_reader:
#         l+=1
#         if l==1:continue
#         relation,start,end=row[0],row[1],row[2]
#         if start==word:
#             rel_entity.append(end)
#     csv_conceptnet.close()
#     f_ent=open(entity, "r")
#     tot_ent = (int)(f_ent.readline())
#     for i in range(tot_ent):
#         content = f_ent.readline()
#         w,e = content.strip().split()
#         if w==word:
#             relation_entity[0]=e
#             continue

#         if w in rel_entity:
#             relation_entity.append(e)
#     f_ent.close()
#     if relation_entity==[]:
#         relation_entity.append('-1')
#     return relation_entity
if __name__=='__main__':
    
    read_example('./dev.json')