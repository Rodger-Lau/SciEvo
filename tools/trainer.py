import torch
import numpy as np
import torch.nn as nn

def train(model,device,train_loader,optimizer,criterion,num_epoch):
    train_loss = []
    running_loss = 0.0
    for epoch in num_epoch:
        for data in train_loader:
            # print(data)
            model.train()
            inputs, real = data
            inputs = inputs.to(device)
            # inputs = inputs.unsqueeze(1)
            real = real.to(device)
            real = real.type(torch.long)
            real = real.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, real)
            # print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss / len(train_loader))
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))