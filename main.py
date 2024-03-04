import datetime
import time
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import transformers
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import SequentialSampler, TensorDataset, RandomSampler, DataLoader
from transformers import logging, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
import numpy as np

logging.set_verbosity_error()

model = BertForSequenceClassification.from_pretrained('D:\pythonProject4\Bert', num_labels=2)
tokenizer = transformers.BertTokenizer.from_pretrained('D:\pythonProject4\Bert')

df = pd.read_csv('D:\pythonProject4\Data\weibo_senti.csv')
label = df['label']
targets = np.array(label).tolist()
targets = torch.unsqueeze(torch.FloatTensor(targets), dim=1)
review = df['review']
sentences = np.array(review).tolist()

# print(sentences[4])
# print(tokenizer.tokenize(sentences[4]))
# print(tokenizer.encode(sentences[4]))
# print(tokenizer.convert_ids_to_tokens(tokenizer.encode(sentences[4])))

#将每一句转成词向量（大于126做截断，小于126做PADDING，加上首尾两个标识，长度总共等于128）
def convert_text_to_token(tokenizer, sentence, limit_size=126):

    tokens = tokenizer.encode(sentence[:limit_size])  #直接截断
    if len(tokens) < limit_size + 2:                  #补齐（pad的索引号就是0）
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens

input_ids = [convert_text_to_token(tokenizer, sen) for sen in sentences]
input_tokens = torch.tensor(input_ids)       #list转为张量

# print(input_tokens.shape)                    #torch.Size([119988, 128])

#建立mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks

atten_masks = attention_masks(input_ids)
attention_tokens = torch.tensor(atten_masks)

#数据分割
train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_tokens, targets, random_state=666, test_size=0.2)
train_masks, test_masks, _, _ = train_test_split(attention_tokens, input_tokens, random_state=666, test_size=0.2)

# print(train_inputs.shape, test_inputs.shape)      #torch.Size([95990, 128]) torch.Size([23998, 128])
# print(train_masks.shape)                          #torch.Size([95990, 128])和train_inputs形状一样
# print(train_inputs[0])
# print(train_masks[0])

#构造迭代器
BATCH_SIZE=6
LEARNING_RATE=2e-5
EPSILON = 1e-8

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

# for i, (train, mask, label) in enumerate(train_dataloader):
#      print(train.shape, mask.shape, label.shape)               #torch.Size([6, 128]) torch.Size([6, 128]) torch.Size([6, 1])
#      break
# print('len(train_dataloader)=', len(train_dataloader))

device = torch.device("cuda:0")
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)
loss = nn.CrossEntropyLoss()

#学习率预热
epochs = 5
#training steps 的数量: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs
# 设计 learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

#计算准确率
def binary_acc(preds, labels):
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()
    acc = correct.sum().item() / len(correct)
    return acc

#计算训练时间
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

#模型训练
def train(model, optimizer):
    t0 = time.time()
    avg_loss, avg_acc = [], []
    with open("log.txt", "w") as f2:
        model.train()
        for step, batch in enumerate(train_dataloader):
            # 每隔40个batch 输出一下所用时间.
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    time: {:}.'.format(step, len(train_dataloader), elapsed))
                f2.write('  Batch {:>5,}  of  {:>5,}.    time: {:}.'.format(step, len(train_dataloader), elapsed))
                f2.write('\n')
                f2.flush()
            b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss, logits = output[0], output[1]
            avg_loss.append(loss.item())
            acc = binary_acc(logits, b_labels)
            avg_acc.append(acc)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)      #大于1的梯度将其设为1.0, 以防梯度爆炸
            optimizer.step()              #更新模型参数
            scheduler.step()              #更新learning rate
        avg_acc = np.array(avg_acc).mean()
        avg_loss = np.array(avg_loss).mean()
        return avg_loss, avg_acc

#模型评估
def evaluate(model):
    avg_acc = []
    model.eval()         #表示进入测试模式

    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            acc = binary_acc(output[0], b_labels)
            avg_acc.append(acc)
    avg_acc = np.array(avg_acc).mean()
    return avg_acc

#训练及评估模型
best_acc = 0
with open("acc.txt", "w") as f:
    for epoch in range(epochs):
        train_loss, train_acc = train(model, optimizer)
        print('epoch={},训练准确率={}，损失={}'.format(epoch+1, train_acc, train_loss))
        test_acc = evaluate(model)
        print("epoch={},测试准确率={}".format(epoch+1, test_acc))
        # 将每次测试结果实时写入acc.txt文件中
        f.write("epoch=%03d,Accuracy= %.3f" % (epoch + 1, test_acc))
        f.write('\n')
        f.flush()
        # 记录最佳测试分类准确率并写入best_acc.txt文件中
        if test_acc > best_acc:
            f3 = open("best_acc.txt", "w")
            f3.write("epoch=%d,best_acc= %.3f" % (epoch + 1, test_acc))
            f3.close()
            best_acc = test_acc
            torch.save(model, "model.pth")
            print("模型保存成功")