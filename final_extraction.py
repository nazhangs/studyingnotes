import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from nltk import pos_tag
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree
from nltk.tree import Tree
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.probability import FreqDist

from nltk.metrics.scores import accuracy
from nltk.metrics.scores import precision
from nltk.metrics.scores import recall
from nltk.metrics.scores import f_measure

os.environ.get('CLASSPATH')
style.use('fivethirtyeight')

def process_text():
    corpus_root = "/Users/zhangna/Downloads/tm/CCAT"
    raw_text = PlaintextCorpusReader(corpus_root, '.*')
    stop_words = stopwords.words('english')
    token_text = raw_text.words()
    token_text = [w for w in token_text if not w in stop_words]
    return token_text

def stanford_tagger(token_text):
    st = StanfordNERTagger('/Users/zhangna/Downloads/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz','/Users/zhangna/Downloads/stanford-ner/stanford-ner.jar',encoding='utf-8')
    ne_tagged=st.tag(token_text)
    return(ne_tagged)

def bio_tagger(ne_tagged):
    bio_tagged = []
    prev_tag = "O"
    for token, tag in ne_tagged:
        if tag == "O":
            bio_tagged.append((token, tag))
            prev_tag = tag
            continue
        if tag != "O" and prev_tag == "O":
            bio_tagged.append((token, "B-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag == tag:
            bio_tagged.append((token, "I-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag != tag:
            bio_tagged.append((token, "B-"+tag))
            prev_tag = tag
    return bio_tagged

def stanford_tree(bio_tagged):
    tokens, ne_tags = zip(*bio_tagged)
    list_tokens=list(tokens)
    for i in range(len(list_tokens)):
        punc_gra=re.match(r'.+[0-9a-zA-Z]',list_tokens[i])
        if punc_gra:
            pass
        elif list_tokens != ".":
            list_tokens[i]=","

    pos_tags = [ pos for token, pos in pos_tag(list_tokens)]
    
    conlltags = [(token, pos, ne) for token, pos, ne in zip(tokens, pos_tags, ne_tags)]

    return conlltags

def conlltree(conlltags):
    ne_tree = conlltags2tree(conlltags)
    return ne_tree

def structure_ne(ne_tree):
    ne = []
    for subtree in ne_tree:
        if type(subtree) == Tree:
            ne_label = subtree.label()
            ne_string = " ".join([token for token, pos in subtree.leaves()])
            ne.append((ne_string, ne_label))
    return ne

def org_extraction(ne_list):
    organizations = []
    for i in range(len(ne_list)):
        if ne_list[i][1] == "ORGANIZATION":
            organizations.append((ne_list[i][0],ne_list[i][1]))
    fdist1 = FreqDist(organizations)
    top5 = fdist1.most_common(5) 
    return top5

def relation_extraction(main_list,top5):
    SVO_list=[]

    for token, tag_pos, tag_bio in main_list:
        verbs_gra = re.match(r'VB+',tag_pos)
        org_gra = re.match(r'\w\-ORGANIZATION',tag_bio)
        if token == "." :
            SVO_list.append('\n')
            continue
        if tag_bio != "O":
            SVO_list.append((token,tag_pos,tag_bio))
        if tag_bio == "O" and verbs_gra:
            SVO_list.append((token,tag_pos,tag_bio))
   
    ob_list = []
    str_ob = ""
    pre_token=""
    for i in SVO_list:
        
        if i != '\n':
            verbs_gra = re.match(r'VB+',i[1])
            org_gra = re.match(r'\w\-ORGANIZATION',i[2])
            per_gra = re.match(r'\w\-PERSON',i[2])
            loc_gra = re.match(r'\w\-LOCATION',i[2])
            bio_gra = re.match(r'^B',i[2])
            if verbs_gra:
                if pre_token=="ACT" or pre_token=="":    
                    str_ob = str_ob+"[ACT: "+i[0]+"],"
                    pre_token = "ACT"
                else:
                    str_ob = str_ob+"],"+"[ACT: "+i[0]+"],"
                    pre_token = "ACT"
            elif org_gra:
                if bio_gra:
                    if pre_token!="" and pre_token!="ACT":
                        str_ob = str_ob+"],"+"[ORG: "+i[0]
                        pre_token="ORG"
                    else:
                        str_ob = str_ob+"[ORG: "+i[0] 
                        pre_token="ORG"
                else:
                    str_ob = str_ob+" "+i[0]
                    pre_token="ORG"
            elif per_gra:
                if bio_gra:
                    if pre_token!="" and pre_token!="ACT":
                        str_ob = str_ob+"],"+"[PER: "+i[0]
                        pre_token="PER"
                    else:
                        str_ob = str_ob+"[PER: "+i[0] 
                        pre_token="PER"
                else:
                    str_ob = str_ob+" "+i[0]
                    pre_token="PER"
            elif loc_gra:
                if bio_gra:
                    if pre_token!="" and pre_token!="ACT":
                        str_ob = str_ob+"],"+"[LOC: "+i[0]
                        pre_token="LOC"
                    else:
                        str_ob = str_ob+"[LOC: "+i[0] 
                        pre_token="LOC"
                else:
                    str_ob = str_ob+" "+i[0]
                    pre_token="LOC"

        else:
            if re.search("ACT",str_ob) :
                if pre_token != "ACT":
                    str_ob=str_ob+"]"
                ob_list.append(str_ob)
                pre_token=""
                str_ob = ""
                
    count=1
    for org in top5:
        print("\n")
        print('#'+str(count)+'  Organization:  '+org[0][0])
        print("|")
        pattern='.+'+org[0][0]+'.+'
        for sentence in ob_list: 
            if re.search(pattern,str(sentence)):
                print("|--------"+sentence) 
                print("|")
        
        count=count+1      
#refernce:https://pythonprogramming.net/testing-stanford-ner-taggers-for-accuracy/  
def evaluation():
    st = StanfordNERTagger('/Users/zhangna/Downloads/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz','/Users/zhangna/Downloads/stanford-ner/stanford-ner.jar',encoding='utf-8')
    raw_annotations=open("/Users/zhangna/test/extraction/test.txt").read()
    split_annotations = raw_annotations.split()
    
    reference_annotations=[]
    for i in range(0,len(split_annotations),2):
        val=split_annotations[i:i+2]
        if len(val) == 2:
            reference_annotations.append(tuple(val))
            
    pure_tokens = split_annotations[::2]
    tagging_perdiction = st.tag(pure_tokens)
    reference_set = set(reference_annotations)
    prediction_set = set(tagging_perdiction)

    
    tagging_accuracy = accuracy(reference_annotations, tagging_perdiction)
    tagging_precision = precision(reference_set, prediction_set)
    tagging_recall = recall(reference_set, prediction_set)
    tagging_fmeasure = f_measure(reference_set, prediction_set)

    return tagging_accuracy,tagging_precision,tagging_recall, tagging_fmeasure

def visualization(tagging_accuracy):
    N = 1
    ind = np.arange(N)
    width = 0.35
    
    fig, ax = plt.subplots()
    
    tagging_percentage = tagging_accuracy * 100
    rects1 = ax.bar(ind, tagging_percentage, width, color='y')
    
    ax.set_xlabel('Extractor')
    ax.set_ylabel('Accuracy(by percentage)')
    ax.set_title('Accuracy of NER Extractor')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('') )
    
    ax.legend(rects1,'Tagger', bbox_to_anchor=(1.05,1),loc=1,borderaxespad=0.)
    plt.show()
if __name__ == '__main__':
    main_list= stanford_tree(bio_tagger(stanford_tagger(process_text())))
    ne_tree = conlltree(main_list)
    ne_list = structure_ne(ne_tree)
    
    top5=org_extraction(ne_list)
    count=1
    
    print("*******************************************")
    print("Top 5 Organizations :")
    for extracted_org in top5:
        print("#"+str(count)+"  "+extracted_org[0][0])
        count=count+1
        
    print ("******************************************")
    relation_extraction(main_list,top5)
    
    tagging_accuracy,tagging_precision,tagging_recall, tagging_fmeasure = evaluation()
    print("***************************")
    print('\n')
    print('Accuracy:  '+str(tagging_accuracy))
    print('Precision:  '+str(tagging_precision))
    print('Recall:  '+str(tagging_recall))
    print('F-measure:  '+str(tagging_fmeasure))
    print('\n')
    print('***************************')
    
    visualization(tagging_accuracy)          
    
