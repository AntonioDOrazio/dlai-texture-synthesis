import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        
        vgg19.requires_grad_(False)

        # Extract only the features part of VGG19
        self.features = vgg19.features.to(DEVICE)
        self.features.requires_grad_(False)
        # Mean and std values for normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

        # Layer indices for style loss
        self.layer_indices = [3, 8, 13, 22, 31]
        print(self.features)


    def forward(self, x):
        # Apply normalization
        x = (x - self.mean) / self.std
        
        features = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.layer_indices:
                features.append(x)
        return features



def gram_matrix(input):
    batch_size, num_features, height, width = input.size()
    features = input.view(batch_size * num_features, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * num_features * height * width)

def style_loss(input, target, reduction="sum"):
    loss = 0
    for i, layer in enumerate(input):
        loss += nn.functional.l1_loss(gram_matrix(layer), gram_matrix(target[i]), reduction=reduction)
    return loss

def content_loss(input, target, reduction="sum"):
    loss = 0
    for i, layer in enumerate(input):
        loss += nn.functional.l1_loss(layer, target[i], reduction=reduction)
    return loss

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(in_features=131072, out_features=512)
        self.output = nn.Linear(in_features=512, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.flatten(x)
        y = self.hidden(y)
        y = self.output(y)
        y = self.sigmoid(y)
        return y