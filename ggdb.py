import torch
from torch import nn
import numpy as np



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(200,2048),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048,18000),
        )
    def forward(self,x):
        return self.layer(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(18000,4096),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            # nn.Linear(4096,1024),
            # nn.ReLU(),
            nn.Linear(4096,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.layer(x)


def Gan_augment_train(original_data=None,generator_model=None,discriminator_model=None,num_epochs=None,optimizer_G=None,optimizer_D=None,device=None,count=None):
    # print("original.shape",original_data.shape)


    generator_model.to(device)
    discriminator_model.to(device)
    generator_model.train()
    discriminator_model.train()
    for epoch in range(num_epochs):
            G_data=np.random.standard_normal(size=(count,200))
            torch_G_data=torch.from_numpy(G_data)
            torch_G_data=torch_G_data.to(torch.float32)
            torch_G_data=torch_G_data.to(device)

            G_paintings=generator_model(torch_G_data)
            noise = np.random.standard_normal(size=(count,90, 200))
            noise = torch.from_numpy(noise)
            noise =noise.view(noise.size(0),-1)
            noise=noise.to(device)
            # print("nois.shape",noise.shape)

            Gauss_nosie=torch.cat([original_data,original_data[0:count-original_data.shape[0],:]],dim=0)
            # print("Gauss_nosie",Gauss_nosie.shape)
            Gauss_nosie=Gauss_nosie.to(device)
            temp=Gauss_nosie+noise
            temp=temp.to(device)
            # print(G_paintings.shape)
            # temp=np.random.standard_normal(size=(count,60,200))
            # temp=torch.from_numpy(temp)
            # temp=temp.view(temp.size(0),-1)
            # # print(temp.shape)
            G_paintings+=temp
            # print(G_paintings.shape)
            # original_data=torch.from_numpy(original_data)
            # original_data=original_data.to(torch.float32)
            # print(original_data.shape)
            # original_data=original_data.view(original_data.size(0),-1)
            # print(original_data.shape)
            original_data=original_data.to(device)
            pro_atrist0=discriminator_model(original_data)
            pro_atrist1=discriminator_model(G_paintings)



            G_loss = -torch.mean(torch.log(pro_atrist1))
            # Criterion = torch.nn.CrossEntropyLoss()
            Criterion=torch.nn.BCELoss()
            D_loss = Criterion(pro_atrist0, torch.ones_like(pro_atrist0)) + Criterion(pro_atrist1,torch.zeros_like(pro_atrist1))
            # G_loss = -1 / torch.mean(torch.log(1. - pro_atrist1))
            # D_loss = -torch.mean(torch.log(pro_atrist0) + torch.log(1 - pro_atrist1))

            optimizer_G.zero_grad()
            G_loss.backward(retain_graph=True)
            optimizer_D.zero_grad()
            D_loss.backward()

            optimizer_G.step()
            optimizer_D.step()


            print("Train Whole, Epoch: {}/{},G_Loss: {:.4f},D_Loss: {:.4f}".format(
                (epoch + 1), num_epochs, G_loss,D_loss
            ))


    return G_paintings
