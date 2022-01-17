import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import time
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path2data = 'C:/Users/327ae/OneDrive/바탕 화면/py/dcgan_dataset/'


h,w = 64,64

transform = transforms.Compose(
    [transforms.Resize((h,w)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
     ]
)
train_ds = datasets.STL10(path2data,split='train',download=True,transform=transform)

batch_size = 64
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

img, label = train_ds[0]
plt.imshow(to_pil_image(0.5*img+0.5))

params = {
    'nz':100,
    'ngf' : 64,
    'ndf':64,
    'img_channel':3,
}

class Generator(nn.Module):
    def __init__(self,params):
        super().__init__()
        nz = params['nz']
        ngf = params['ngf']
        img_channel = params['img_channel']
        
        self.dconv1 = nn.ConvTranspose2d(nz,ngf*8,4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*8)
        self.dconv2 = nn.ConvTranspose2d(ngf*8,ngf*4, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*4)
        self.dconv3 = nn.ConvTranspose2d(ngf*4,ngf*2,4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*2)
        self.dconv4 = nn.ConvTranspose2d(ngf*2,ngf,4,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.dconv5 = nn.ConvTranspose2d(ngf,img_channel,4,stride=2,padding=1,bias=False)
        
    def forward(self,x):
        x = F.relu(self.bn1(self.dconv1(x)))
        x = F.relu(self.bn2(self.dconv2(x)))
        x = F.relu(self.bn3(self.dconv3(x)))
        x = F.relu(self.bn4(self.dconv4(x)))
        x = torch.tanh(self.dconv5(x))
        return x

x = torch.randn(1,100,1,1, device=device)
model_gen = Generator(params).to(device)
out_gen = model_gen(x)

class Discriminator(nn.Module):
    def __init__(self,params):
        super().__init__()
        img_channel = params['img_channel'] # 3
        ndf = params['ndf'] # 64

        self.conv1 = nn.Conv2d(img_channel,ndf,4,stride=2,padding=1,bias=False)
        self.conv2 = nn.Conv2d(ndf,ndf*2,4,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(ndf*2)
        self.conv3 = nn.Conv2d(ndf*2,ndf*4,4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(ndf*4)
        self.conv4 = nn.Conv2d(ndf*4,ndf*8,4,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(ndf*8)
        self.conv5 = nn.Conv2d(ndf*8,1,4,stride=1,padding=0,bias=False)

    def forward(self,x):
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)),0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)),0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)),0.2)
        x = torch.sigmoid(self.conv5(x))
        return x.view(-1,1)

# check
x = torch.randn(16,3,64,64,device=device)
model_dis = Discriminator(params).to(device)
out_dis = model_dis(x)

def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

# 가중치 초기화 적용
model_gen.apply(initialize_weights);
model_dis.apply(initialize_weights);

# 손실 함수 정의
loss_func = nn.BCELoss()

# 최적화 함수
from torch import optim
lr = 0.0003
beta1 = 0.5
beta2 = 0.999

opt_dis = optim.Adam(model_dis.parameters(),lr=lr,betas=(beta1,beta2))
opt_gen = optim.Adam(model_gen.parameters(),lr=lr,betas=(beta1,beta2))

model_gen.train()
model_dis.train()

batch_count=0
num_epochs=100
start_time = time.time()
nz = params['nz'] # 노이즈 수 100
loss_hist = {'dis':[],
             'gen':[]}

# for epoch in range(num_epochs):
#     for xb, yb in train_dl:
#         ba_si = xb.shape[0]

#         xb = xb.to(device)
#         yb_real = torch.ones(ba_si,1).to(device)
#         yb_fake = torch.zeros(ba_si,1).to(device)

#         # generator
#         model_gen.zero_grad()

#         z = torch.randn(ba_si,nz,1,1).to(device) # noise
#         out_gen = model_gen(z) # 가짜 이미지 생성
#         out_dis = model_dis(out_gen) # 가짜 이미지 식별

#         g_loss = loss_func(out_dis,yb_real)
#         g_loss.backward()
#         opt_gen.step()

#         # discriminator
#         model_dis.zero_grad()
        
#         out_dis = model_dis(xb) # 진짜 이미지 식별
#         loss_real = loss_func(out_dis,yb_real)

#         out_dis = model_dis(out_gen.detach()) # 가짜 이미지 식별
#         loss_fake = loss_func(out_dis,yb_fake)

#         d_loss = (loss_real + loss_fake) / 2
#         d_loss.backward()
#         opt_dis.step()

#         loss_hist['gen'].append(g_loss.item())
#         loss_hist['dis'].append(d_loss.item())

#         batch_count += 1
#         if batch_count % 100 == 0:
#             print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' %(epoch, g_loss.item(), d_loss.item(), (time.time()-start_time)/60))
            
plt.figure(figsize=(10,5))
plt.title('Loss Progress')
plt.plot(loss_hist['gen'], label='Gen. Loss')
plt.plot(loss_hist['dis'], label='Dis. Loss')
plt.xlabel('batch count')
plt.ylabel('Loss')
plt.legend()
plt.show()

path2models = 'C:/Users/327ae/OneDrive/바탕 화면/py/dcgan_dataset/dcgan_model/'

path2weights_gen = os.path.join(path2models, 'weights_gen.pt')
path2weights_dis = os.path.join(path2models, 'weights_dis.pt')

torch.save(model_gen.state_dict(), path2weights_gen)
torch.save(model_dis.state_dict(), path2weights_dis)

weights = torch.load(path2weights_gen)
model_gen.load_state_dict(weights)

model_gen.eval()

# fake image 생성
with torch.no_grad():
    fixed_noise = torch.randn(16, 100,1,1, device=device)
    label = torch.randint(0,10,(16,), device=device)
    img_fake = model_gen(fixed_noise).detach().cpu()

plt.figure(figsize=(10,10))
plt.subplot(4,4,2+1)
plt.imshow(to_pil_image(0.5*img_fake[2]+0.5), cmap='gray')
plt.axis('off')
