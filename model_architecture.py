import torch
import torch.nn.functional as F
from torch import nn

import matplotlib.pyplot as plt
def visualize_self_similarity_map(similarity_scores):
    """
    Visualize the self-similarity map using a heatmap.
    
    Args:
        similarity_scores (torch.Tensor): The similarity scores sp,q of size (H, W).
    """
    similarity_scores_np = similarity_scores.permute(0,2,3,1).detach().cpu().numpy()
    
    for score in similarity_scores_np:

        # Create a heatmap using matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(score, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Self-Similarity Map')
        plt.xlabel('Horizontal Shift (q)')
        plt.ylabel('Vertical Shift (p)')
        plt.show()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn.init as init
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)



'''
    Encoder block
'''
import torchvision
class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True).to(DEVICE)
        
        vgg19.requires_grad_(False)
        
        # Extract only the features part of VGG19
        self.features = vgg19.features.to(DEVICE)
        self.features.requires_grad_(False)
        self.features.eval()
        # Mean and std values for normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

        # Layer indices for style loss
        self.layer_indices = [3, 8, 17, 26, 35]

        self.convolution_layer = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1).to(DEVICE)



    def forward(self, x):
        x=x.to(DEVICE)
        x = (x - self.mean) / self.std
        
        features = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.layer_indices:
                features.append(x)
    
        y_5_2 = self.convolution_layer(features[4])
        y_4_2 = features[3]
        y_3_2 = features[2]
        return y_5_2, y_4_2, y_3_2



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # TODO nn.SyncBatchNorm
        self.conv1 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=3, out_channels=64, kernel_size=3, stride=1), 
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv2_1 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=64, out_channels=128, kernel_size=3, stride=2), 
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv2_2 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=128, out_channels=128, kernel_size=3, stride=1), 
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv3_1 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=128, out_channels=256, kernel_size=3, stride=2), 
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.conv3_2 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=256, out_channels=256, kernel_size=3, stride=1), 
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.conv4_1 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=256, out_channels=512, kernel_size=3, stride=2), 
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.conv4_2 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=512, out_channels=512, kernel_size=3, stride=1), 
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.conv5_1 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=512, out_channels=1024, kernel_size=3, stride=2), 
            nn.BatchNorm2d(1024),
            nn.ReLU())
        self.conv5_2 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=1024, out_channels=1024, kernel_size=3, stride=1), 
            nn.BatchNorm2d(1024),
            nn.ReLU())

        self.apply(weights_init)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2_1(y)
        y = self.conv2_2(y)
        y = self.conv3_1(y)
        y_3_2 = self.conv3_2(y)
        y = self.conv4_1(y_3_2)
        y_4_2 = self.conv4_2(y)
        y = self.conv5_1(y_4_2)
        y_5_2 = self.conv5_2(y)

        return y_5_2, y_4_2, y_3_2



'''
    Self similarity block
'''
class SelfSimilarity(nn.Module):
    def __init__(self):
        super(SelfSimilarity, self).__init__()

    def calculate_similarity_score(self, input, p, q):
        """
        Calculate the similarity score s(p, q) for a given (p, q)-shifting.

        Args:
            input (torch.Tensor): The original feature map of size (B, C, H, W).
            p (int): The vertical shift.
            q (int): The horizontal shift.

        Returns:
            torch.Tensor: The similarity score s(p, q).
        """
        B, C, H, W = input.shape
        m_min = max(0, p)
        m_max = min(p + H, H)
        n_min = max(0, q)
        n_max = min(q + W, W)

        numerator = torch.norm(input[:, :, m_min:m_max, n_min:n_max] - input[:, :, m_min - p:m_max - p, n_min - q:n_max - q], p=2)
        denominator = torch.norm(input)
        
        return -numerator / denominator

    def compute_similarity_scores(self, input):
        """
        Compute similarity scores sp,q for all possible shifted copies.

        Args:
            input (torch.Tensor): The original feature map of size (B, C, H, W).

        Returns:
            torch.Tensor: The similarity scores sp,q of size (H + 1, W + 1).
        """
        B, C, H, W = input.shape
        similarity_scores = torch.zeros((B, H + 1, W + 1), dtype=input.dtype).to(input.device)

        for p in range(-H // 2, H // 2 + 1):  # Adjust the range here
            for q in range(-W // 2, W // 2 + 1):  # Adjust the range here
                similarity_scores[:, p + H // 2, q + W // 2] = self.calculate_similarity_score(input, p, q)

        return similarity_scores

    def aggregate_shifted_features(self, input, similarity_scores):
        """
        Aggregate the shifted copies of F using the similarity scores.

        Args:
            input (torch.Tensor): The original feature map of size (B, C, H, W).
            similarity_scores (torch.Tensor): The similarity scores for all possible shifted copies.

        Returns:
            torch.Tensor: The aggregated feature G of size (C, H + 1, W + 1).
        """
        B, C, H, W = input.shape
        Csum = torch.zeros((B, C, H + 1, W + 1), dtype=input.dtype).to(input.device)

        for i in range(H + 1):
            for j in range(W + 1):
                p = i - H // 2
                q = j - W // 2
                weight = similarity_scores[:, p + H // 2, q + W // 2]

                Csum[:, :, i, j] = input[:, :, i % H, j % W] * weight[:, None, None]  # Proper broadcasting

        Csum = torch.mean(Csum, dim=1, keepdim=True)

        return Csum

    def forward(self, x):
        similarity_scores = self.compute_similarity_scores(x)
        y = self.aggregate_shifted_features(x, similarity_scores)
        return y


import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfSimilarityMap(nn.Module):
    def __init__(self):
        super(SelfSimilarityMap, self).__init__()


    def generate_map(self, batch_size, height, width, channels):
        center_h_start = height // 2
        center_h_end = center_h_start + height
        center_w_start = width // 2
        center_w_end = center_w_start + width
        
        map = torch.zeros(batch_size, channels, 2 * height, 2 * width)
        map[:, :, center_h_start:center_h_end, center_w_start:center_w_end] = 1
        

        return map

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # First term
        x_ups = F.upsample_bilinear(x, x.shape[3]+1)
        x_squared = x_ups.pow(2)
        conv_filter1 = torch.ones(1, channels, 1, 1).to(DEVICE)
        term_1 = F.conv2d(x_squared, conv_filter1, padding=0)

        # Second term
        term_1_min = torch.min(term_1.reshape((x.shape[0],-1)),dim=1)[0].view(x.shape[0], 1, 1, 1).expand(term_1.shape)
        term_1_max = torch.max(term_1.reshape((x.shape[0],-1)),dim=1)[0].view(x.shape[0], 1, 1, 1).expand(term_1.shape)

        padding = (width // 2, width // 2, height // 2, height // 2)

        x_padded = F.pad(x, padding)
        conv_filter2 = x

        for i in range(x.shape[0]):
            if i == 0:
                term_2 = F.conv2d(x_padded[i], conv_filter2[i].unsqueeze(0), padding=0).unsqueeze(0)
            else:
                term_2 = torch.cat((term_2, 
                                   F.conv2d(x_padded[i], conv_filter2[i].unsqueeze(0), padding=0).unsqueeze(0)),
                                   dim=0)


        # Third term
        x_map = self.generate_map(x.shape[0], height, width, channels).to(DEVICE)
        conv_filter3 = x

        for i in range(x.shape[0]):
            if i == 0:
                term_3 = F.conv2d(x_map[i], conv_filter3[i].unsqueeze(0), padding=0).unsqueeze(0)

            else:
                term_3 = torch.cat((term_3, 
                                   F.conv2d(x_map[i], conv_filter3[i].unsqueeze(0), padding=0).unsqueeze(0)),
                                   dim=0)


        numerator = term_1 - 2*term_2 + term_3
        
        similarity_map = -numerator/(term_1+1e-8)

        similarity_map_min = torch.min(similarity_map.reshape((x.shape[0],-1)),dim=1)[0].view(x.shape[0], 1, 1, 1).expand(similarity_map.shape)
        similarity_map_max = torch.max(similarity_map.reshape((x.shape[0],-1)),dim=1)[0].view(x.shape[0], 1, 1, 1).expand(similarity_map.shape)


        similarity_map = (similarity_map-similarity_map_min)/(similarity_map_max-similarity_map_min)

        return similarity_map




'''
    Transposed convolution blocks 3,4,5
'''
class TransConvBlock(nn.Module):

    def __init__(self, input_size, filter_size = 256, div_factor=4, in_features=None):
        super(TransConvBlock, self).__init__()
        self.filterBranch_conv1 = nn.Sequential(
            nn.Conv2d(filter_size, filter_size, 3, 1, padding=1), 
            nn.ReLU()
        )

        self.input_size = input_size
        self.filter_size = filter_size
        self.div_factor = div_factor

        # Transposed Conv Filter Weight
        self.filterBranch_conv2 = nn.Conv2d(filter_size, filter_size, 3, 1, padding=1)

        # Transposed Conv FIlter Bias
        self.avg_pooling = nn.AvgPool2d(kernel_size=3)

        dummy_input = torch.randn(in_features).unsqueeze(0)

        fc1_in_features = self.avg_pooling(self.filterBranch_conv2(self.filterBranch_conv1(dummy_input))).reshape(-1).shape[0]

        self.filterBranch_FC1 = nn.Linear(out_features=filter_size, in_features=fc1_in_features)

        # Compute self similarity
        self.self_similarity = SelfSimilarityMap().to(DEVICE)

        self.selfSimilarityMapBranch_Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.selfSimilarityMapBranch_Conv2 = nn.Conv2d(8,1,3,1, padding=1)
        # Transposed conv
        #self.transposed_conv = nn.ConvTranspose2d(in_channels=1, out_channels=filter_size, kernel_size=(input_size // div_factor, input_size // div_factor))
        
        self.outputBranch_conv = nn.Sequential(
            nn.Conv2d(filter_size, filter_size, 3, stride=1, padding=1), nn.ReLU())
        

        self.apply(weights_init)

    def forward(self, x, raw_image=None):

        # Filters
        w = self.filterBranch_conv1(x)
        w = self.filterBranch_conv2(w)
        #print("WEIGHTS ", w.shape)

        b = self.avg_pooling(w)
        #print("SHAPE B ", b.shape)
        b = self.filterBranch_FC1(b.reshape(b.shape[0], -1))
        #print("BIASES ", b.shape)

        # Self similarity
        sim_in = x
        if raw_image is not None:
            sim_in = torch.nn.functional.upsample_bilinear(raw_image, x.shape[3])

        y = self.self_similarity(sim_in)
        #visualize_self_similarity_map(y)
        #print("similarity size ", y.shape)

        #print("SELF SIM ", y.shape)
        y = self.selfSimilarityMapBranch_Conv1(y)
        #print("IS NAN ", torch.isnan(y))
        y = self.selfSimilarityMapBranch_Conv2(y)
        #visualize_self_similarity_map(y)

        for i in range(y.shape[0]):
            if i == 0:
                output = nn.functional.conv_transpose2d(input=y[i].unsqueeze(0), weight=w[i].unsqueeze(0), bias=b[i])#, kernel_size=(self.input_size // self.div_factor, self.input_size // self.div_factor))
            else:
                output = torch.cat((output, 
                                   nn.functional.conv_transpose2d(input=y[i].unsqueeze(0), weight=w[i].unsqueeze(0), bias=b[i])),
                                   dim=0)
        
        output = self.outputBranch_conv(output)
        return output


'''
    Decoder block
'''
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2) # block 5 output #1024
        self.conv6 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=1024, out_channels=512, kernel_size=3, stride=1), 
            nn.BatchNorm2d(512),
            nn.ReLU())
        # sum
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv7 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=512, out_channels=256, kernel_size=3, stride=1), 
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv8 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=256, out_channels=128, kernel_size=3, stride=1), 
            nn.BatchNorm2d(128),
            nn.ReLU())        
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv9 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=128, out_channels=64, kernel_size=3, stride=1), 
            nn.BatchNorm2d(64),
            nn.ReLU())       
        self.conv10 = nn.Conv2d(padding=1, in_channels=64, out_channels=3, kernel_size=3, stride=1) 
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)


    def forward(self, block5, block4, block3):
        y = self.upsample1(block5)
        y = self.conv6(y)

        #print("Summing ", y.shape, " with ", block4.shape)

        y = y + block4#self.sigmoid(y + block4)
        y = self.upsample2(y)
        y = self.conv7(y)
        y = y + block3 #self.sigmoid(y + block3)
        y = self.upsample3(y)
        y = self.conv8(y)
        y = self.upsample4(y)
        y = self.conv9(y)
        y = self.conv10(y)
        y = self.sigmoid(y)
        return y
    