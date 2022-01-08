import os

import numpy as np

import pandas as pd


import torch
import torch.optim as optim

for dirname, _, filenames in os.walk('C:/Users/327ae/OneDrive/바탕 화면/py/2021-ai-w3-p1'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv('C:/Users/327ae/OneDrive/바탕 화면/py/2021-ai-w3-p1/train.csv')

test = pd.read_csv('C:/Users/327ae/OneDrive/바탕 화면/py/2021-ai-w3-p1/test.csv')


torch.manual_seed(1)

x_train = torch.Tensor(np.array(train)[:,1:-1])
y_train = torch.Tensor(np.array(train)[:,-1])
x_test = torch.Tensor(np.array(test)[:,1:])


W = torch.zeros((4,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.001)

nb_epochs = 1000

for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # Matrix 연산!!
    hypothesis = x_train.matmul(W) + b # or .mm or @
    
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

predict = x_test.matmul(W) + b

submit = pd.read_csv('C:/Users/327ae/OneDrive/바탕 화면/py/2021-ai-w3-p1/submit_sample.csv')

submit['Expected'] = predict.detach().numpy() 

print(submit)
