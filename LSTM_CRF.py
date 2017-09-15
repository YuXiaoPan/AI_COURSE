
# coding: utf-8

# In[56]:

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import random
import numpy as np

class Lang():
    def __init__(self, name):
        self.name = name
        self.word_to_index = {}
        self.index_to_word = {}
    
    def addword(self, word):
        if not word in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
            
    def addembds(self):
        if self.name == 'hinglish':
            self.embds = torch.eye(len(self.word_to_index), len(self.word_to_index))
        else:
            self.embds = torch.eye(len(self.word_to_index), len(self.word_to_index)).type(torch.LongTensor)
    
    def getembds(self, word):
        if self.name == 'hinglish':
            idx = torch.LongTensor([self.word_to_index[char] for char in word])
        else:
            idx = torch.LongTensor([self.word_to_index[word]])
        return torch.index_select(self.embds, 0, idx)

hinglish = Lang('hinglish') # 印度英语
hindi = Lang('hindi')# 印度语
datas = [] # hinglish_hindi_pair
with open('./hinglish_data/hinglish_hindi_pair_sample', 'r') as f:
    for line in f.readlines():
        a, b = line.strip().split('\t')
        a = a.strip()
        b = b.strip()
        datas.append((a, b))
        for char in a:
            hinglish.addword(char)
        hindi.addword(b)
hinglish.addembds()
hindi.addembds()

def timestamp():
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())


# In[57]:

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import random
import numpy as np
import math

class LSTM_CRF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(LSTM_CRF, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers=2, dropout=0.5)
        self.hidden2output = nn.Linear(self.hidden_size, self.output_size)
        
        self.hidden = self.init_hidden()
        self.f = f
        
    def init_hidden(self):
        return [Variable(torch.zeros(2, 1, self.hidden_size)).cuda(),
               Variable(torch.zeros(2,1,self.hidden_size)).cuda()]
    
    def _lstm_out(self, hinglish):
        hinglish = hinglish.view(len(hinglish), 1, -1)
        self.hidden = self.init_hidden()
        output, hidden = self.rnn(hinglish, self.hidden)
        output = output.view(len(hinglish), -1)
        output = self.hidden2output(output)
        return output
    
    def log_sum_exp(self, vec):
        max_score, idx = torch.max(vec, 1)
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score +             torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    
    def target_score(self,feats, target_index):
        score = self.f[target_index]
        for i in range(len(feats)):
            score = score + feats[i,target_index ]  / (i+1)
#         score = score + torch.sum(feats[:,target_index])
        return score
    
    
    def forward_score(self, feats):
        forward = self.f
        for i in range(len(feats)):
            forward = forward + feats[i] / (i+1)
#         forward = forward + torch.sum(feats, 0)
        score = self.log_sum_exp(forward.view(1, -1))
        return score
    
    def viterbi(self, feats):
        output = self.f
#         for i in range(len(feats)):
#             output = output + feats[i] / (i+1)
        output = output+ torch.sum(feats, 0) 
        return output.view(1, -1)
    
    # target_index:hindi的下标
    # init_prob:hindi的词频
    def _train(self, hinglish, target_index):
        feats = self._lstm_out(hinglish)
        gold_score = self.target_score(feats,target_index)
        f_score = self.forward_score(feats)
        return f_score - gold_score
    
    def forward(self, hinglish):
        feats = self._lstm_out(hinglish)
        output = self.viterbi(feats)
        return F.log_softmax(output)
        


# In[58]:

import torch.optim as optim
import time
import os
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

epochs = 5000 * 1000
model_path = './model_crf/'
fw = open('./log_crf_'+timestamp()+'.txt', 'w', encoding='utf-8')
if not os.path.exists(model_path):
    os.makedirs(model_path)



rnn = LSTM_CRF(len(hinglish.word_to_index), 128, len(hindi.word_to_index), Variable(torch.zeros(len(hindi.word_to_index))).cuda())
rnn.load_state_dict(torch.load('./model_crf/rnn_3366_0.013.m'))
rnn=rnn.cuda()
dtype = torch.cuda.FloatTensor
# rnn = torch.load('./model/rnn_2017-09-02_00-33-11.m')


def write(msg):
    print(msg)
    fw.write(msg)
    fw.write('\n')
    fw.flush()


    
def eval():
    rnn.eval()
    total = 0
    c1 = 0
    c3 = 0
    c3_1 = 0
    for data in datas:
        rnn.hidden = rnn.init_hidden()
#         if random.randint(0, 4) >= 3:continue
        total += 1
        src, dst = data
        src_vec = Variable(hinglish.getembds(src)).cuda()
#         limit = int(len(src_vec)/1.2) if eff else len(src_vec)
        dst_id = hindi.word_to_index[dst]
        output = rnn(src_vec)
        
        _, idx = torch.topk(output, 1)
        if dst_id in idx.data.view(-1):
            c1 += 1
        _, idx = torch.topk(output, 3)
        if dst_id in idx.data.view(-1):
            c3 += 1
        
        rnn.hidden = rnn.init_hidden()
        limit = int(len(src_vec)/1.2)
        output = rnn(src_vec[0:limit])
        _, idx = torch.topk(output, 3)
        if dst_id in idx.data.view(-1):
            c3_1 += 1
        
    res1 = "topk:{} correct:{}/{} rate:{}\n".format(1, c1, total, c1/total)
    res3 = "topk:{} correct:{}/{} rate:{}\n".format(3, c3, total, c3/total)
    res3_1 = "topk:{} correct:{}/{} rate:{}".format(3, c3_1, total, c3_1/total)
    rnn.train()
    write(res1+res3+res3_1)
    return c3_1

def early_stoping(res, item, max_to_keep=1):
    assert len(res) <= max_to_keep, 'early_stopping 列表越界'
    if len(res) < max_to_keep:
        res.append(item)
        return False
    else:
        res.append(item)
        idx = np.argmin(res)
        if idx == 0:
            res.pop(0)
            return True
        else:
            res.pop(0)
            return False
score = 1000000000.
loss_function = nn.NLLLoss()
lr = 0.1
optimizer = optim.SGD(rnn.parameters(), lr=lr)
res = []

time_start = time.time()

eval()

max_c3 = 0
pre_loss = 10000
best_lr = None
for epoch in range(epochs):
    data = datas[epoch % len(datas)]
    src, dst = data
    rnn.zero_grad()
    src_vec = Variable(hinglish.getembds(src)).cuda()
    loss = rnn._train(src_vec, hindi.word_to_index[dst])
    loss.backward()
    optimizer.step()

    score += loss.data[0]
    
    if epoch % 5000==4999:
        write(("{}%=>loss:{}".format(int(epoch/epochs * 100), score/ 5000)))
        c3_1 = eval()
        
        if c3_1 >= max_c3:
            max_c3 = c3_1
            best_lr = str(lr)[:7]
            torch.save(rnn.state_dict(), model_path+'rnn_'+str(max_c3)+'_'+best_lr+".m")
            pre_loss = score/5000
        elif c3_1 < max_c3 and pre_loss < score/5000:
            rnn.load_state_dict(torch.load('./model_crf/rnn_'+str(max_c3)+'_'+best_lr+'.m'))
            lr = lr * 0.8
            print("==>lr:"+str(lr))
            if(lr < 1e-5):break
            optimizer = optim.SGD(rnn.parameters(), lr=lr)
            pre_loss = 10000
        else:
            pre_loss = score/5000
        
        time_end = time.time()
        print("time:"+str(time_end-time_start))
        time_start = time.time()
        
        score = 0

