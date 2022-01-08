import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
torch.manual_seed(1)

train = pd.read_csv("C:/Users/327ae/OneDrive/바탕 화면/py/2021-ai-w4-p1/train.csv")
test = pd.read_csv("C:/Users/327ae/OneDrive/바탕 화면/py/2021-ai-w4-p1/test.csv")
train.sample(5)

from sklearn.preprocessing import RobustScaler
sc_X = RobustScaler()

X_train = pd.DataFrame(sc_X.fit_transform(train.drop(["Diabetes"], axis = 1)))
X_test = pd.DataFrame(sc_X.transform(test))
y_train = train.Diabetes

X_train = torch.FloatTensor(np.array(X_train))
y_train = torch.FloatTensor(np.array(y_train))
X_test = torch.FloatTensor(np.array(X_test))

epochs = 3000

W = torch.zeros((8,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

loss = torch.nn.BCEWithLogitsLoss()

optimizer = optim.SGD([W,b],lr = 0.001)

for epoch in range(epochs+1):
    h = torch.sigmoid(X_train.matmul(W)+b)
    
    cost = loss(X_train.matmul(W) + b, y_train.unsqueeze(1))
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, epochs, cost.item()))
        
        
hypothesis = torch.sigmoid(X_test.matmul(W) + b)
prediction = (hypothesis >= torch.FloatTensor([0.5])).type(torch.uint8) 

submit = pd.read_csv("C:/Users/327ae/OneDrive/바탕 화면/py/2021-ai-w4-p1/submit.csv")
for i, value in enumerate(prediction):
    submit["Diabetes"][i] = value.item()


submit.to_csv("output.csv", index = False)

print(submit)
