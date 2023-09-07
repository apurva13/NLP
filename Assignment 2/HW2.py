#!/usr/bin/env python
# coding: utf-8

## Importing all the required packages
import sys
import json
from collections import defaultdict, Counter
from numpy import log
import copy

## Reading the test data

test_data=[]
filePath="test.txt"
        
with open(filePath, "r") as file:
    for x in file:
        x=x.rstrip()
        test_data.append(x.split("\t"))

print(len(test_data))
# print(test_data)


## Reading the train data
train_data=[]
filePath="dev.txt"
        
with open(filePath, "r") as file:
    for x in file:
        x=x.rstrip()
        train_data.append(x.split("\t"))


## Reading the dev data
dev_data=[]
filePath="dev.txt"
        
with open(filePath, "r") as file:
    for x in file:
        x=x.rstrip()
        dev_data.append(x.split("\t"))

print(len(dev_data))
# print(dev_data)

## Fetching the words list from the dev data

dev_words=list()
temp=[]
for i in dev_data:
    
    if len(i)<2:
        dev_words.append(temp)
        temp=[]
    else:
        temp.append(i[1])
        
print(len(dev_words))
# print(dev_words)


test_words=list()
temp2=[]
for i in test_data:
    
    if len(i)<2:
        test_words.append(temp2)
        temp2=[]
    else:
        temp2.append(i[1])
        
len(test_words)

## Fetching the words and tag combination list from the dev data for accuracy

dev_words2=list()
temp=[]
for i in dev_data:
    if len(i)<2:
        dev_words2.append(temp)
        temp=[]
    else:
        temp.append(i[1]+'/'+i[2])

print(len(dev_words2))
# print(dev_words2)

## Fetching all the tags from dev_data in the list

dev_tags=list()
for each in dev_data:
    if len(each)>1:
        dev_tags.append(each[2])


print(len(dev_tags))
# print(dev_tags)

## Fetching all the 45 unique tags from the train data and storing it into a set

POS_total=set()
for each in train_data:
    if len(each)>1:
        POS_total.add(each[2])
    

print(len(POS_total))
# print(POS_total)

## Calculating the frequency of each words in the train data

train_count={}#defaultdict(lambda: defaultdict(int))
train_count2={}#defaultdict(lambda: defaultdict(int))

for lines in train_data:
        if len(lines)>1:
            if lines[1] not in train_count:
                train_count[lines[1]]=1
            else:
                train_count[lines[1]]+=1
                
                
print(len(train_count))
# print(train_count)


## Calulating the frequency of unknown words that are below the threshold (3)

count=0
unknown_words={}
for word, val in train_count.items():
    if val >= 3:
        train_count2[word]=val
    else:
        unknown_words[word]=0
        count+=val
unk={"unk":count}
# print(count)

## Creating a sorted dictionary in which the unknown words count is at first and then the sorted words

train_count3={}
train_count3["unk"]=count
train_count2=dict(sorted(train_count2.items(),key=lambda x:-x[1]))
for k, v in train_count2.items():
    train_count3[k]=v
# print(train_count3)


## Creating a text file named vocab which shows word,index and occurences


with open('vocab.txt', 'w') as vocab:
    i=1
    for k,v in train_count3.items():
        vocab.write('%s\t%s\t%s\n' % (k, i, v))
        i+=1
        


## Creating a transition and emmision parameters in HMM

transition_matrix=defaultdict(lambda: defaultdict(int))
emmision_matrix=defaultdict(lambda: defaultdict(int))

for w in train_data:
    if len(w)>1:
        previous="start"
        term, pos = w[1],w[2]
        if (term.isdigit()):
            emmision_matrix[pos]['<digit>'] +=1
        if term in unknown_words.keys():
            emmision_matrix[pos]['<unknown>'] +=1

        emmision_matrix[pos][term] +=1
        transition_matrix[previous][pos] +=1

        previous=pos

        pos='fin'

        transition_matrix[previous][pos] += 1
    
    
transition_matrix['fin']={}
    
for previous in transition_matrix:
    for present in transition_matrix:
        transition_matrix[previous][present]=transition_matrix[previous].get(present,0)+1
    


emm_prob={}
trans_prob={}

#Calculating the Emission Probability
for pos in emmision_matrix:
    emm_prob[pos] = dict()
    Count_pos_emm = sum(emmision_matrix[pos].values())
    for term in emmision_matrix[pos]:
        emm_prob[pos][term] = emmision_matrix[pos][term]/Count_pos_emm


#Calculating the Transition Probability
for pos in transition_matrix:
    trans_prob[pos] = dict()
    Count_pos_trans = sum(transition_matrix[pos].values())
    for term in transition_matrix[pos]:
        trans_prob[pos][term] = transition_matrix[pos][term]/Count_pos_trans
        


transition={}
for i in trans_prob:
    for j in trans_prob[i]:
        transition[str(i)+','+str(j)]=trans_prob[i][j]


emission={}
for i in emm_prob:
    for j in emm_prob[i]:
        emission[str(i)+','+str(j)]=emm_prob[i][j]



hmm={"Emmision:":emission,"Transition:":transition}



get_ipython().system('pip install simplejson')
import json as simplejson


json_data = json.dumps(hmm)
jsonDataFile = open("hmm.json", "w")
jsonDataFile.write(simplejson.dumps(simplejson.loads(json_data), indent=4, sort_keys=True))
jsonDataFile.close()


## Greedy Algorithm running on dev_data for accuracy calculation

pred_pos =[]
pred_term=[]
curr_pos=''
arr=[]

hsh={}
i=0
for term in dev_data:
    #print(len(term))
    if term==".":
        i=0
        arr.append(".")
        i=i+1
        continue
    elif len(term)>1:
        t=term[1]
        if i >=1:
            for pos in POS_total:
                prob_t=trans_prob[curr_pos][pos]
                prob_e=emm_prob[pos].get(t,0.00000001)
                total_prob=prob_t*prob_e
                pred_pos.append([total_prob,pos])
            pred_pos.sort(key=lambda x: -x[0])
            curr_pos=pred_pos[0][1]
            pred_pos=[]
            
        else:
            for pos in POS_total:
                prob_t=trans_prob['start'][pos]
                prob_e=emm_prob[pos].get(t,0.00000001)
                total_prob=prob_t*prob_e
                pred_pos.append([total_prob,pos])
            pred_pos.sort(key=lambda x: -x[0])
            curr_pos=pred_pos[0][1]
            pred_pos=[]
        i=i+1
        arr.append(curr_pos)



from sklearn.metrics import accuracy_score
print(accuracy_score(dev_tags, arr))

## Greedy algorithm where we check the maximum prob for each word with all 45 unique tags and pick the maximum prob


pred_pos2 =[]
pred_term2=[]
curr_pos2=''
arr2=[]

hsh={}
i=0
for term in test_data:
    #print(len(term))
    if term==".":
        i=0
        arr2.append(".")
        i=i+1
        continue
    elif len(term)>1:
        t=term[1]
        if i >=1:
            for pos in POS_total:
                prob_t=trans_prob[curr_pos][pos]
                prob_e=emm_prob[pos].get(t,0.00000001)
                total_prob=prob_t*prob_e
                pred_pos2.append([total_prob,pos])
            pred_pos2.sort(key=lambda x: -x[0])
            curr_pos2=pred_pos2[0][1]
            pred_pos2=[]
            
        else:
            for pos in POS_total:
                prob_t=trans_prob['start'][pos]
                prob_e=emm_prob[pos].get(t,0.00000001)
                total_prob=prob_t*prob_e
                pred_pos2.append([total_prob,pos])
            pred_pos2.sort(key=lambda x: -x[0])
            curr_pos2=pred_pos2[0][1]
            pred_pos2=[]
        i=i+1
        arr2.append((term[0],term[1],curr_pos2))


with open ('greedy.out','w') as file:
    for i in arr2:
        if len(i)>1:
            file.write('%s\t%s\t%s\n' % (i[0],i[1],i[2]))


##Calculating viterbi Algorithm

negProb = -1000000
output = list()
for index,sentences in enumerate(dev_words):
    tagProb, tagProbBack = list(), list()
    for wordIndex, currentWord in enumerate(sentences):
        tagProb.append({})
        tagProbBack.append({})

        if(wordIndex !=0 and currentWord.isdigit()):
            for tag in POS_total:
                tagProb[wordIndex][tag] = negProb
                currentEmissionProb = log(emm_prob[tag].get('<digit>',1e-9))

                for prevTag in tagProbBack[wordIndex-1]:
                    jointProb = currentEmissionProb+trans_prob[prevTag][tag]
                    if(tagProb[wordIndex][tag] < jointProb):
                        tagProb[wordIndex][tag] = jointProb
                        tagProbBack[wordIndex][tag] = prevTag

        elif(wordIndex == 0):
            for tag in POS_total:
                tagProb[wordIndex][tag] = log(emm_prob[tag].get(currentWord,1e-9))+log(trans_prob['start'].get(tag))
                tagProbBack[wordIndex][tag] = 'start'

        else:
            unknownWords = 1
            for tag in tagProb[wordIndex-1]:
                tagProb[wordIndex][tag] = negProb
                currentEmissionProb = emm_prob[tag].get(currentWord,0)

                if(currentEmissionProb != 0):
                    currentEmissionProb = log(currentEmissionProb)
                    unknownWords = 0

                    for prevTag in tagProbBack[wordIndex-1]:
                        jointProb = log(trans_prob[prevTag].get(tag))+currentEmissionProb+tagProb[wordIndex-1][prevTag]

                        if(tagProb[wordIndex][tag] < jointProb):
                            tagProb[wordIndex][tag] = jointProb
                            tagProbBack[wordIndex][tag] = prevTag

            if(unknownWords):
                for tag in emm_prob:
                    for prevTag in tagProbBack[wordIndex-1]:
                        jointProb = log(emm_prob[tag].get('<unknown>',1e-9)) + log(trans_prob[prevTag].get(tag)) + tagProb[wordIndex-1][prevTag]

                        if(tagProb[wordIndex][tag] < jointProb):
                            tagProb[wordIndex][tag] = jointProb
                            tagProbBack[wordIndex][tag] = prevTag


    for tag in tagProbBack[wordIndex]:
        endProb = log(trans_prob[tag].get('fin'))
        tagProb[wordIndex][tag] += endProb


    highProbTag = max(tagProb[-1], key=tagProb[-1].get)

    taggedSentence = list()
    for i in range(len(tagProb)-1, -1, -1):
        wordTag = sentences[i]+'/'+highProbTag
        taggedSentence.insert(0, wordTag)
        highProbTag = tagProbBack[i][highProbTag]

    output.append(taggedSentence)


##Accuracy calculation for viterbi


data1=[]
for i in range(len(dev_words2)):
    for x in dev_words2[i]:
        i=x.split('/')
        data1.append(i)
data2=[]
for i in range(len(output)):
    for x in output[i]:
        i=x.split('/')
        data2.append(i)
correct=0
incorrect=0
for i in range(len(data2)):
    if data2[i][0]==data1[i][0]:
        if data2[i][1]==data1[i][1]:
            correct=correct+1
        else:
            incorrect=incorrect+1
t=correct+incorrect
print(correct/t, incorrect/t)


##Calculating viterbi Algorithm for test_data


negProb = -1000000
output2 = list()
for index,sentences in enumerate(test_words):
    tagProb, tagProbBack = list(), list()
    for wordIndex, currentWord in enumerate(sentences):
        tagProb.append({})
        tagProbBack.append({})

        if(wordIndex !=0 and currentWord.isdigit()):
            for tag in POS_total:
                tagProb[wordIndex][tag] = negProb
                currentEmissionProb = log(emm_prob[tag].get('<digit>',1e-9))

                for prevTag in tagProbBack[wordIndex-1]:
                    jointProb = currentEmissionProb+trans_prob[prevTag][tag]
                    if(tagProb[wordIndex][tag] < jointProb):
                        tagProb[wordIndex][tag] = jointProb
                        tagProbBack[wordIndex][tag] = prevTag

        elif(wordIndex == 0):
            for tag in POS_total:
                tagProb[wordIndex][tag] = log(emm_prob[tag].get(currentWord,1e-9))+log(trans_prob['start'].get(tag))
                tagProbBack[wordIndex][tag] = 'start'

        else:
            unknownWords = 1
            for tag in tagProb[wordIndex-1]:
                tagProb[wordIndex][tag] = negProb
                currentEmissionProb = emm_prob[tag].get(currentWord,0)

                if(currentEmissionProb != 0):
                    currentEmissionProb = log(currentEmissionProb)
                    unknownWords = 0

                    for prevTag in tagProbBack[wordIndex-1]:
                        jointProb = log(trans_prob[prevTag].get(tag))+currentEmissionProb+tagProb[wordIndex-1][prevTag]

                        if(tagProb[wordIndex][tag] < jointProb):
                            tagProb[wordIndex][tag] = jointProb
                            tagProbBack[wordIndex][tag] = prevTag

            if(unknownWords):
                for tag in emm_prob:
                    for prevTag in tagProbBack[wordIndex-1]:
                        jointProb = log(emm_prob[tag].get('<unknown>',1e-9)) + log(trans_prob[prevTag].get(tag)) + tagProb[wordIndex-1][prevTag]

                        if(tagProb[wordIndex][tag] < jointProb):
                            tagProb[wordIndex][tag] = jointProb
                            tagProbBack[wordIndex][tag] = prevTag


    for tag in tagProbBack[wordIndex]:
        endProb = log(trans_prob[tag].get('fin'))
        tagProb[wordIndex][tag] += endProb


    highProbTag = max(tagProb[-1], key=tagProb[-1].get)

    taggedSentence = list()
    for i in range(len(tagProb)-1, -1, -1):
        wordTag = sentences[i]+'/'+highProbTag
        taggedSentence.insert(0, wordTag)
        highProbTag = tagProbBack[i][highProbTag]

    output2.append(taggedSentence)



test_vit_out=[]
for i in output2:
    ind=1
    t=[]
    for j in i:
        t.append((ind,j.split('/')[0],j.split('/')[1]))
        ind=ind+1
    test_vit_out.append(t)
    


with open ('viterbi.out','w') as file:
    for i in test_vit_out:
        for j in i:
            if len(i)>1:
                file.write('%s\t%s\t%s\n' % (j[0],j[1],j[2]))




