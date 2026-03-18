---
title: ML-Autoencoder
date: 2024-06-06 22:10:03
author: sheldon
categories: 
- ML
tags:
- ML
---

We will build a complete autoencoder pipeline and compare the results with PCA on dimensionality reduction tasks.

```python
# load necessary libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from time import time
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
```

`Fashion-MNIST` is a dataset of Zalando’s article images consisting of of 60,000 training examples and 10,000 test examples. Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

```python
# download FashionMNIST dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor() # ToTensor() transforms the data to tensor type and rescale [0,255] uint8 to [0,1] float
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

nsamples = 10
imgs = training_data.data[0:nsamples,:]
imgs = np.expand_dims(imgs,1)
imgs = torch.tensor(imgs)
imgs = imgs/255
print(imgs.shape)
labels = training_data.train_labels[0:nsamples]
classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fig=plt.figure(figsize=(20,5),facecolor='w')
for i in range(nsamples):
    ax = plt.subplot(1,nsamples, i+1)
    plt.imshow(imgs[i, 0, :, :], cmap=plt.get_cmap('gray'))
    ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# print(dir(training_data)) # print all attribute of an object
# print(training_data.data[0:9,:]) # print info of the first picture
# plt.imshow(training_data.data[0,:], cmap=plt.get_cmap('gray')) # visualize the first picture
```

```txt
torch.Size([10, 1, 28, 28])
```

{% asset_img Fashion-MNIST.png %}

## Step 1: Prepare Data

```python
batch_size = 128

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# check the first batch of the dataset
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape) # 
    print("Shape of y: ", y.shape, y.dtype)
    break
    
# N: number of data instance in a batch
# C: channel, number of colors in a pixel here
# [H, W]: Height and width of a picture
```

```txt
Shape of X [N, C, H, W]:  torch.Size([128, 1, 28, 28])
Shape of y:  torch.Size([128]) torch.int64
```

## Step 2: Define Neural Nets

```python
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.ReLU()
        )

        self.decoder_stack = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder_stack(x)
        x = self.decoder_stack(z)
        return x

model = AutoEncoder()
print(model)
```

```txt
AutoEncoder(
  (encoder_stack): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=784, out_features=128, bias=True)
    (2): ReLU()
    (3): Linear(in_features=128, out_features=16, bias=True)
    (4): ReLU()
    (5): Linear(in_features=16, out_features=2, bias=True)
    (6): ReLU()
  )
  (decoder_stack): Sequential(
    (0): Linear(in_features=2, out_features=16, bias=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=128, bias=True)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=784, bias=True)
    (5): Sigmoid()
  )
)
```

## Step 3: Define loss function and the optimizer

```python
loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters())
```

## Step 4: Train the neural nets

```python
# epochs: how many times we would like to traverse the whole dataset
epochs=50

for i in range(epochs): # over epochs
    tic = time()
    model.train()
    train_loss = 0
    for j, (X, y) in enumerate(train_dataloader): # over batches 
        # Compute prediction error
        pred = model(X)

        loss = loss_fn(pred, X.reshape(X.size(0),-1))
        train_loss += loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print learning process
#         if j % 100 == 0:
#             loss, current = loss.item(), j * len(X)
#             print(f"epoch {i} batch {j} loss: {loss/batch_size:>7f}")
    train_time = time() - tic
    
    # print test results after every epochs    

    print(f"epoch {i} training time: {train_time:>7f}s, train loss: {train_loss/len(train_dataloader.dataset):>7f}")    
```

```txt
epoch 0 training time: 5.038011s, train loss: 0.000559
epoch 1 training time: 5.130287s, train loss: 0.000480
epoch 2 training time: 5.422238s, train loss: 0.000448
epoch 3 training time: 5.880179s, train loss: 0.000387
epoch 4 training time: 5.521627s, train loss: 0.000376
epoch 5 training time: 5.126778s, train loss: 0.000367
epoch 6 training time: 5.206013s, train loss: 0.000362
epoch 7 training time: 5.712925s, train loss: 0.000358
epoch 8 training time: 5.457368s, train loss: 0.000355
epoch 9 training time: 5.184317s, train loss: 0.000349
epoch 10 training time: 5.153089s, train loss: 0.000344
epoch 11 training time: 5.222293s, train loss: 0.000343
epoch 12 training time: 5.173838s, train loss: 0.000340
epoch 13 training time: 5.678778s, train loss: 0.000338
epoch 14 training time: 5.614803s, train loss: 0.000336
epoch 15 training time: 5.425945s, train loss: 0.000334
epoch 16 training time: 5.560995s, train loss: 0.000332
epoch 17 training time: 5.370713s, train loss: 0.000332
epoch 18 training time: 5.385829s, train loss: 0.000330
epoch 19 training time: 5.537868s, train loss: 0.000330
epoch 20 training time: 5.746025s, train loss: 0.000328
epoch 21 training time: 5.276327s, train loss: 0.000329
epoch 22 training time: 5.439416s, train loss: 0.000326
epoch 23 training time: 5.563997s, train loss: 0.000324
epoch 24 training time: 5.372393s, train loss: 0.000323
...
epoch 46 training time: 5.110753s, train loss: 0.000297
epoch 47 training time: 5.206847s, train loss: 0.000271
epoch 48 training time: 5.143513s, train loss: 0.000262
epoch 49 training time: 5.325914s, train loss: 0.000255
```

## Step 5: Dimensionality reduction using PCA

```python
# retrieve image data as array for PCA, each image [28,28] become a vector [1, 784]
X = training_data.data.reshape(60000,-1).numpy()
X = X/X.max()

# fit PCA 
pca = PCA(n_components=2)
pca.fit(X)
print(np.cumsum(pca.explained_variance_ratio_))

# transform sampled imgs using PCA and recover images
imgs_pca = imgs.reshape(nsamples,-1).numpy() # reshape sample images to vectors
imgs_reduced = pca.transform(imgs_pca)
imgs_recovered = pca.inverse_transform(imgs_reduced)
imgs_recovered = np.reshape(imgs_recovered, [nsamples,28,28]) # reshape vectors to sample images
```

```txt
[0.29039228 0.46794538]
```

## Step 6: Visual comparison between autoencoder and PCA

```python
fig=plt.figure(figsize=(20,5),facecolor='w')
for i in range(nsamples):
    ax = plt.subplot(1,nsamples, i+1)
    plt.imshow(imgs[i, 0, :, :], cmap=plt.get_cmap('gray'))
    ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

fig=plt.figure(figsize=(20,5),facecolor='w')
pred = model(imgs)
pred = pred.reshape(pred.size(0),1,28,28).detach().numpy()
for i in range(nsamples):
    ax = plt.subplot(1,nsamples, i+1)
#     print(pred[i, 0, :, :])
    plt.imshow(pred[i, 0, :, :], cmap=plt.get_cmap('gray'))
    ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

fig=plt.figure(figsize=(20,5),facecolor='w')
for i in range(nsamples):
    ax = plt.subplot(1,nsamples, i+1)
    plt.imshow(imgs_recovered[i, :, :], cmap=plt.get_cmap('gray'))
    ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

{% asset_img original.png %}

{% asset_img autoencoder.png %}

{% asset_img PCA.png %}