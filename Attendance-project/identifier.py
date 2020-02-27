import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

data_transforms = {
    'train':transforms.Compose(
    [transforms.RandomRotation(10),
     transforms.RandomHorizontalFlip(0.5),
     transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.3),
     transforms.RandomResizedCrop(size=256, scale=(0.85,1.0)),
     transforms.RandomPerspective(distortion_scale=0.3),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
     transforms.RandomApply([AddGaussianNoise()],p=0.8),
    ]
),
    'valid': transforms.Compose(
    [transforms.RandomRotation(10),
     transforms.RandomHorizontalFlip(0.5),
     transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.3),
     transforms.RandomPerspective(distortion_scale=0.3),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
     transforms.RandomApply([AddGaussianNoise()], p=0.8),
    ]
)
}

data_dir = 'Computer club members'
image_datasets = {x :datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x])
                  for x in ['train', 'valid']}

data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                          shuffle=True, num_workers=4)
                for x in ['train', 'valid']}

dataset_sizes = {x:len(image_datasets[x]) for x in ['train', 'valid']}

class_names = image_datasets['train'].classes

device = torch.device('cpu')

model = models.resnet18(pretrained=True)

#model = models.vgg19()

# for param in model.parameters():
#     param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 12)
model.to(device)
loss_fn = nn.CrossEntropyLoss()

opti = optim.Adam(model.parameters(), lr=0.00001)
exp_lr_scheduler = lr_scheduler.StepLR(opti,step_size=10,gamma=0.85)

def train_model(model, loss_fn, optimizer, scheduler, num_epochs=25):
    #best_model_weights = copy.deepcopy(model.state_dict())
    #best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, names in data_loaders[phase]:
                with torch.set_grad_enabled(phase=='train'):
                    inputs = inputs.to(device)
                    names = names.to(device)
                    optimizer.zero_grad()

                    output = model(inputs)
                    _, preds = torch.max(output, 1)
                    loss = loss_fn(output, names)
                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                running_loss+=loss.item()*inputs.size(0)
                running_corrects+=torch.sum(preds==names.data)
            if phase=='train':
                scheduler.step()
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects.double()/dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    return model

model = train_model(model,loss_fn, scheduler=exp_lr_scheduler, optimizer=opti, num_epochs=130)

torch.save(model, './computer_club.pth')
