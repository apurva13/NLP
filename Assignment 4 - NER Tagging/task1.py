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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


## Reading the train data
train_data=[]
dev_data=[]
filePath="train"
        
with open(filePath, "r") as file:
    for x in file:
        x=x.rstrip()
        train_data.append(x.split(" "))
        
with open('dev', "r") as file:
    for x in file:
        x=x.rstrip()
        dev_data.append(x.split(" "))
        
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


# In[3]:


dev_words_tag_list=list()
temp=[]
for i in dev_data:
    if len(i)<2:
        dev_words_tag_list.append(temp)
        temp=[]
    else:
        temp.append((i[0],i[1],i[2]))


# In[4]:


#Defining Hyperparameters
embedding_dim = 100
lstm_hidden_dim = 256
lstm_layers = 1
lstm_dropout = 0.33
linear_output_dim = 128
batch_size = 16
learning_rate = 0.9
num_epochs = 20


# In[5]:


train_sentences=[]
train_tags=[]

with open('train', 'r', encoding='utf-8') as f:
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
            
dev_sentences=[]
dev_tags=[]

with open('dev', 'r', encoding='utf-8') as f:
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


# In[6]:



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
dev_loader = data.DataLoader(dev_dataset, batch_size=batch_size )#, drop_last=True)

# Define the number of words and classes
num_words = len(train_dataset.train_word_idx) + 1
num_classes = len(train_dataset.train_label_idx) + 1


# In[7]:


get_ipython().run_cell_magic('time', '', 'import torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom sklearn.metrics import f1_score, accuracy_score\n\n\n\n# Define the model architecture\nclass BLSTM(nn.Module):\n    def __init__(self, embedding_dim, lstm_hidden_dim, lstm_layers, lstm_dropout, linear_output_dim):\n        super(BLSTM, self).__init__()\n        self.embedding_dim = embedding_dim\n        self.lstm_hidden_dim = lstm_hidden_dim\n        self.lstm_layers = lstm_layers\n        self.lstm_dropout = lstm_dropout\n        self.linear_output_dim = linear_output_dim\n\n        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=self.embedding_dim)\n        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.lstm_hidden_dim, num_layers=self.lstm_layers, dropout=self.lstm_dropout, bidirectional=True)\n        self.linear = nn.Linear(self.lstm_hidden_dim*2, self.linear_output_dim)\n        self.activation = nn.ELU()\n        self.classifier = nn.Linear(self.linear_output_dim, num_classes)\n\n    def forward(self, x):\n        x = self.embedding(x)\n        x, _ = self.lstm(x)\n        x = self.linear(x)\n        x = self.activation(x)\n        x = self.classifier(x)\n        return x\n    \n#     def init_weights(self):\n#         # to initialize all parameters from normal distribution\n#         # helps with converging during training\n#         for name, param in self.named_parameters():\n#               nn.init.normal_(param.data, mean=0, std=0.1)')


# In[8]:


idx_to_label={}
for key,val in train_label_idx.items():
    idx_to_label[val]=key


# In[9]:


def calculate_metrics(true_labels, predicted_labels):
    """
    Calculates precision, recall and F1 score for the given predictions and targets.
    """
    # Flatten the predictions and targets
    predicted_labels = np.array(predicted_labels)
    predictions_flat = predicted_labels.flatten()
    true_labels = np.array(true_labels)
    targets_flat = true_labels.flatten()

    # Calculate the number of true positives, false positives and false negatives
    tp = ((predictions_flat == 1) & (targets_flat == 1)).sum().item()
    fp = ((predictions_flat == 1) & (targets_flat == 0)).sum().item()
    fn = ((predictions_flat == 0) & (targets_flat == 1)).sum().item()

    # Calculate precision, recall and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


# In[10]:


# Define the training loop
        
def train(model, train_loader, loss_fn, optimizer, scheduler, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, num_classes), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step(total_loss/len(train_loader))
        print("Epoch: {}, Loss: {}".format(epoch+1, total_loss/(i+1)))
        
        model.eval()
        dev_loss = 0.0
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dev_loader):
                outputs = model(inputs)
                loss = loss_fn(outputs.view(-1, num_classes), targets.view(-1))
                dev_loss += loss.item()
                scheduler.step(dev_loss/len(dev_loader))
                true_labels.extend(targets.cpu().numpy())
                predicted_labels.extend(torch.argmax(outputs, axis=-1).cpu().numpy())
    
    
    with open('Predicted.txt', 'w') as pred:
        xP=[]
        for pred_tags in predicted_labels:
            t=[]
            for i in pred_tags:
                t.append(idx_to_label[i])
                
            xP.append(t)
        for inp in xP:
            pred.write(' '.join(inp)+'\n')      

    # Calculate metrics on dev set
    precision, recall, f1_score = calculate_metrics(true_labels, predicted_labels)


# In[11]:


pad_idx=train_label_idx['<PAD>']
pad_idx


# In[12]:


get_ipython().run_cell_magic('time', '', '# Train the model\nmodel = BLSTM(embedding_dim, lstm_hidden_dim, lstm_layers, lstm_dropout, linear_output_dim)\n# Define the loss function and optimizer\nloss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)\noptimizer = optim.SGD(model.parameters(), lr=learning_rate)\nscheduler = StepLR(optimizer, step_size=2, gamma=0.1)\ntrain(model, train_loader, loss_fn, optimizer,scheduler, 20)')


# In[13]:


pred=[]
with open("Predicted.txt", "r") as f:
    for x in f:
        pred.append(x.split(" "))


# In[14]:


leng=[len(i) for i in dev_words_tag_list ]


# In[15]:


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


# In[16]:


final_output = {}
idx = 0
for i in range(len(dev_words_tag_list)):
    for j in range(len(dev_words_tag_list[i])):
        final_output[idx] = (dev_words_tag_list[i][j][0], dev_words_tag_list[i][j][1],dev_words_tag_list[i][j][2], tg[i][j])
        idx += 1
    
print(final_output)


# In[17]:


check=0
with open("perl_check_dev2.txt", 'w') as f: 
    for key,i in final_output.items() : 
        if i[0] == '1' and check!=0:
            f.write('\n')
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
        else:
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
            check=check+1


# In[18]:


get_ipython().system("perl conll03eval < {'perl_check_dev2.txt'} #step_size=2, gamma=0.1 and 20 epochs")


# In[19]:


torch.save(model,'blstm1.pt')


# In[ ]:




