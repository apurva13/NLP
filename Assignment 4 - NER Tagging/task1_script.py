#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from torch.optim.lr_scheduler import StepLR
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ### Reading and Processing the data

# In[ ]:


file1 = sys.argv[1]
file2 = sys.argv[2]
file3= sys.argv[3]
file4= sys.argv[4]


# In[2]:


## Reading the train, dev and test data
train_data=[]
dev_data=[]
test_data=[]
filePath=file1
        
with open(filePath, "r") as file:
    for x in file:
        x=x.rstrip()
        train_data.append(x.split(" "))
        
with open(file2, "r") as file:
    for x in file:
        x=x.rstrip()
        dev_data.append(x.split(" "))

with open(file3, "r") as file:
    for x in file:
        x=x.rstrip()
        test_data.append(x.split(" "))
        
train_words=list()
temp=[]
for i in train_data:
    if len(i)<2:
        train_words.append(temp)
        temp=[]
    else:
        temp.append(i[1])
        
t_words=set()
for i in train_data:
    if len(i)>1:
        t_words.add(i[1])
        
t_tags=set()
for i in train_data:
    if len(i)>1:
        t_tags.add(i[2])
c=2      
train_word_idx={}
for i in t_words:
    train_word_idx[i]=c
    c+=1
    
train_word_idx['<PAD>'] = 0
train_word_idx['<UNK>'] = 1  


train_ner_tag=list()
temp=[]
for i in train_data:
    if len(i)<2:
        train_ner_tag.append(temp)
        temp=[]
    else:
        temp.append(i[2])

        
c=1      
train_label_idx={}
for i in t_tags:
    train_label_idx[i]=c
    c+=1
    
train_label_idx['<PAD>'] = 0


train_sentences=[]
train_tags=[]

with open(file1, 'r', encoding='utf-8') as f:
    words = []
    ner_tags = []
    for line in f:
        if line == '\n':
            train_sentences.append(words)
            train_tags.append(ner_tags)
            words = []
            ner_tags = []
        else:
            items = line.strip().split()
            word = items[1]
            ner_tag = items[2]
            words.append(word)
            ner_tags.append(ner_tag)
            

dev_words_tag_list=list()
temp=[]
for i in dev_data:
    if len(i)<2:
        dev_words_tag_list.append(temp)
        temp=[]
    else:
        temp.append((i[0],i[1],i[2]))
        
dev_sentences=[]
dev_tags=[]

with open(file2, 'r', encoding='utf-8') as f:
    words = []
    ner_tags = []
    for line in f:
        if line == '\n':
            dev_sentences.append(words)
            dev_tags.append(ner_tags)
            words = []
            ner_tags = []
        else:
            items = line.strip().split()
            word = items[1]
            ner_tag = items[2]
            words.append(word)
            ner_tags.append(ner_tag)
        
        
test_word_tag_list=[]
for x in test_data:
    if len(x)>1:
        test_word_tag_list.append((x[0],x[1]))
        
        
test_sentences=[]
with open(file3, 'r', encoding='utf-8') as f:
    words = []
    ner_tags = []
    for line in f:
        if line == '\n':
            test_sentences.append(words)
            words = []
        else:
            items = line.strip().split()
            word = items[1]
            words.append(word)
        
test_sentence_tag_list=[]
check=0
tstl_test=[]

for x in test_data:
    if len(x)>1:
            if x[0]=='1':
                check+=1
                if check == 1:
                    tstl_test=[]
                    tstl_test.append((x[0],x[1]))
                elif check==3684:
                    test_sentence_tag_list.append(tstl_test)
                    tstl_test=[]
                    tstl_test.append((x[0],x[1]) )
                    test_sentence_tag_list.append(tstl_test)
                    break
                else: 
                    test_sentence_tag_list.append(tstl_test)
                    tstl_test=[]
                    tstl_test.append((x[0],x[1]))  
            else:
                tstl_test.append((x[0],x[1]))
                
            


# In[3]:


len(test_sentence_tag_list)


# ### Defining Hyperparameters

# In[4]:


embedding_dim = 100
lstm_hidden_dim = 256
lstm_layers = 1
lstm_dropout = 0.33
linear_output_dim = 128
batch_size = 16
learning_rate = 0.9
num_epochs = 20


# ### Defining the dataset and dataloader

# In[5]:


# Define the dataset class
class NERDataset(data.Dataset):
    def __init__(self, sentences, tags,train_word_idx,train_label_idx,max_len):
        self.sentences = sentences
        self.tags = tags
        self.train_word_idx=train_word_idx
        self.train_label_idx=train_label_idx
        self.max_len=max_len

    def __getitem__(self, index):
        sentence=self.sentences[index]
        tags= self.tags[index]
        x = [self.train_word_idx.get(word, 1) for word in self.sentences[index]]
        y = [self.train_label_idx[tag] for tag in tags]
        x = x + [0] * (self.max_len - len(x))  # Pad the sequence with zeros
        y = y + [0] * (self.max_len - len(y))  # Pad the sequence with zeros
        return torch.LongTensor(x), torch.LongTensor(y)
    
    def __len__(self):
        return len(self.sentences)


# Define the train dataset and data loader
train_dataset = NERDataset(train_sentences, train_tags,train_word_idx,train_label_idx,max_len=113)
dev_dataset = NERDataset(dev_sentences, dev_tags,train_word_idx,train_label_idx,max_len=109)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = data.DataLoader(dev_dataset, batch_size=batch_size )
# test_dataset = NERDataset()

# Define the number of words and classes
num_words = len(train_dataset.train_word_idx) + 1
num_classes = len(train_dataset.train_label_idx) + 1


# In[6]:


# Define the dataset class
class NERDataset_test(data.Dataset):
    def __init__(self, sentences,train_word_idx,max_len):
        self.sentences = sentences
        self.train_word_idx=train_word_idx
        self.max_len=max_len

    def __getitem__(self, index):
        sentence=self.sentences[index]
        x = [self.train_word_idx.get(word, 1) for word in self.sentences[index]]
        x = x + [0] * (self.max_len - len(x))  # Pad the sequence with zeros
        return torch.LongTensor(x)
    
    def __len__(self):
        return len(self.sentences)


# Define the train dataset and data loader
test_dataset = NERDataset_test(test_sentences,train_word_idx,max_len=124)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

# Define the number of words and classes
num_words = len(train_dataset.train_word_idx) + 1
num_classes = len(train_dataset.train_label_idx) + 1


# ### Model

# In[7]:




# Define the model architecture
class BLSTM(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim, lstm_layers, lstm_dropout, linear_output_dim):
        super(BLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.linear_output_dim = linear_output_dim

        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.lstm_hidden_dim, num_layers=self.lstm_layers, dropout=self.lstm_dropout, bidirectional=True)
        self.linear = nn.Linear(self.lstm_hidden_dim*2, self.linear_output_dim)
        self.activation = nn.ELU()
        self.classifier = nn.Linear(self.linear_output_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.classifier(x)
        return x


# In[8]:


idx_to_label={}
for key,val in train_label_idx.items():
    idx_to_label[val]=key


# In[11]:


pad_idx=train_label_idx['<PAD>']
pad_idx


# In[ ]:


model=torch.load(file4)


# In[ ]:


model.eval()
predicted_labels = []
with torch.no_grad():
    for i, (inputs, targets) in enumerate(dev_loader):
        outputs = model(inputs)
        predicted_labels.extend(torch.argmax(outputs, axis=-1).cpu().numpy())

with open('dev_predicted.txt', 'w') as pred:
    xP=[]
    for pred_tags in predicted_labels:
        t=[]
        for i in pred_tags:
            t.append(idx_to_label[i])

        xP.append(t)
    for inp in xP:
        pred.write(' '.join(inp)+'\n')

model.eval()
predicted_test_labels = []
with torch.no_grad():
    for i, inputs in enumerate(test_loader):
        outputs = model(inputs)
        predicted_test_labels.extend(torch.argmax(outputs, axis=-1).cpu().numpy())

with open('test_predicted.txt', 'w') as pred:
    xP=[]
    for pred_tags in predicted_test_labels:
        t=[]
        for i in pred_tags:
            t.append(idx_to_label[i])

        xP.append(t)
    for inp in xP:
        pred.write(' '.join(inp)+'\n')


# In[ ]:


pred=[]
with open("dev_predicted.txt", "r") as f:
    for x in f:
        pred.append(x.split(" "))


# In[ ]:


leng=[len(i) for i in dev_words_tag_list ]


# In[ ]:


ik=0
tg=[]
for sent in pred:
    th=[]
    lenk=0
    for jk in sent:
        if lenk<leng[ik]:
            th.append(jk)
            lenk=lenk+1
    tg.append(th)
    ik=ik+1


# In[ ]:


final_output = {}
idx = 0
for i in range(len(dev_words_tag_list)):
    for j in range(len(dev_words_tag_list[i])):
        final_output[idx] = (dev_words_tag_list[i][j][0], dev_words_tag_list[i][j][1],dev_words_tag_list[i][j][2], tg[i][j])
        idx += 1
    
print(final_output)


# In[ ]:


check=0
with open("dev1.out", 'w') as f: 
    for key,i in final_output.items() : 
        if i[0] == '1' and check!=0:
            f.write('\n')
            f.write('%s %s %s\n' % (i[0], i[1],i[3]))
        else:
            f.write('%s %s %s\n' % (i[0], i[1],i[3]))
            check=check+1


# In[ ]:


pred_test=[]
with open("test_predicted.txt", "r") as f:
    for x in f:
        pred_test.append(x.split(" "))


# In[ ]:


leng_test=[len(i) for i in test_sentence_tag_list ]


# In[ ]:


c=0
new=[]
for sent in pred_test:
    temp=[]
    lenk=0
    for jk in sent:
        if lenk<leng_test[c]:
            temp.append(jk)
            lenk=lenk+1
    new.append(temp)
    c+=1


# In[ ]:


test_word_tag_list[1]


# In[ ]:


final_output_test = {}
idx_test = 0
for i in range(len(test_sentence_tag_list)-1):
    for j in range(len(test_sentence_tag_list[i])):
        final_output_test[idx_test] = (test_sentence_tag_list[i][j][0], test_sentence_tag_list[i][j][1], new[i][j])
        idx_test += 1


# In[ ]:


new[0][0]


# In[ ]:


check=0
with open("test1.out", 'w') as f: 
    for key,i in final_output_test.items() : 
        if i[0] == '1' and check!=0:
            f.write('\n')
            f.write('%s %s %s\n' % (i[0], i[1],i[2]))
        else:
            f.write('%s %s %s\n' % (i[0], i[1],i[2]))
            check=check+1


# In[ ]:




