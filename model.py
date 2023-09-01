
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from vgg_loss import VGGFeatures, style_loss, content_loss
from model_architecture import Encoder, TransConvBlock, Decoder, VGGEncoder
from torch import optim
import pytorch_lightning as L
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.optim import lr_scheduler
from sliced_loss.sliced_loss import slicing_loss
from torchmetrics.image import StructuralSimilarityIndexMeasure



_ = torch.manual_seed(123)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextureSynthesizer(L.LightningModule):
    def __init__(self, width, height, train=False, encoder_type="default"):
        super().__init__()
        if encoder_type == "default": 
            self.encoder = Encoder()
        elif encoder_type == "vgg":
            self.encoder = VGGEncoder()
        else:
            raise TypeError("Encoder type: [default, vgg]")


        dummy_input = torch.randn(1, 3, width, height)
        y_5_2, y_4_2, y_3_2 = self.encoder(dummy_input)

        self.transconv_block3 = TransConvBlock(input_size=256, filter_size=256, div_factor=4, in_features=(y_3_2.squeeze(0).shape))
        self.transconv_block4 = TransConvBlock(input_size=256, filter_size=512, div_factor=8, in_features=(y_4_2.squeeze(0).shape))
        self.transconv_block5 = TransConvBlock(input_size=256, filter_size=1024, div_factor=16, in_features=(y_5_2.squeeze(0).shape))
        self.decoder = Decoder()

        if train==True:
            self.vgg = VGGFeatures()
            self.vgg.requires_grad_(False)
            self.vgg.train(False)
            self.vgg.eval()
            self.fid = FrechetInceptionDistance(feature=64, normalize=True).to(DEVICE)
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)


    def encode(self, x):
        y_5_2, y_4_2, y_3_2 = self.encoder(x)
        return y_5_2, y_4_2, y_3_2
    
    def compute_similarity(self, x_5_2, x_4_2, x_3_2):
        sim_5_2, _  = self.self_similarity(x_5_2)
        sim_4_2, _  = self.self_similarity(x_4_2) 
        sim_3_2, _  = self.self_similarity(x_3_2) 

        return sim_5_2, sim_4_2, sim_3_2

    def forward(self, x):
        # weights for the transpConv
        y_5_2, y_4_2, y_3_2 = self.encode(x)

        y_3_2 = self.transconv_block3(y_3_2)#, x)
        y_4_2 = self.transconv_block4(y_4_2)#, x)
        y_5_2 = self.transconv_block5(y_5_2)#, x)


        image = self.decoder(y_5_2, y_4_2, y_3_2)

        return image
    
    def configure_optimizers(self, lr, step_size, gamma):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)


    def training_step(self, batch, batch_idx, style_loss_type="gram"):
        self.fid.reset()

        sample_images = batch[0].to(DEVICE)
        target_images = batch[1].to(DEVICE)


        sample_reconstructed = self.forward(sample_images)

        reconstructed_features = self.vgg(sample_reconstructed)
        target_features = self.vgg(target_images)

        alpha = 0.05 
        beta = 120
        gamma = 0.2e4
        
        content_loss_value = alpha * content_loss(reconstructed_features[3], target_features[3], reduction="sum")

        if style_loss_type == "gram":
            style_loss_value =  beta * style_loss(reconstructed_features, target_features, reduction="sum")
        elif style_loss_type == "wasserstein":
            beta = beta*1e3
            style_loss_value = beta * slicing_loss(sample_reconstructed, target_images)
        else: 
            raise TypeError("style_loss_type: [gram, wasserstein]")


        self.fid.update(sample_reconstructed, real = False)
        self.fid.update(target_images, real= True)
        fid_loss_value = gamma * self.fid.compute()
        loss = content_loss_value + style_loss_value + fid_loss_value

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {"train/content_loss": content_loss_value, "train/style_loss": style_loss_value, "train/fid_loss": fid_loss_value, "train/loss": loss}

        reconstructed = sample_reconstructed.detach()

        del sample_images
        del target_images
        del sample_reconstructed
        del reconstructed_features
        del target_features

        with torch.no_grad():
            torch.cuda.empty_cache()

        return reconstructed, metrics

    def validation_step(self, batch, batch_idx, style_loss_type="gram"):
        
        with torch.no_grad():

            self.fid.reset()

            sample_images = batch[0].to(DEVICE)
            target_images = batch[1].to(DEVICE)

            sample_reconstructed = self.forward(sample_images)

            reconstructed_features = self.vgg(sample_reconstructed)
            target_features = self.vgg(target_images)

            alpha = 0.05 
            beta = 120
            gamma = 0.2e4
            
            content_loss_value = alpha * content_loss(reconstructed_features[3], target_features[3], reduction="sum")
            

            if style_loss_type == "gram":
                style_loss_value =  beta * style_loss(reconstructed_features, target_features, reduction="sum")
            elif style_loss_type == "wasserstein":
                beta = beta*1e3
                style_loss_value = beta * slicing_loss(sample_reconstructed, target_images)
            else: 
                raise TypeError("style_loss_type: [gram, wasserstein]")

            self.fid.update(sample_reconstructed, real = False)
            self.fid.update(target_images, real= True)
            metric_fid = self.fid.compute()
            fid_loss_value = gamma * metric_fid
            loss = content_loss_value + style_loss_value + fid_loss_value

            ssim = self.ssim(sample_reconstructed, target_images)

            metrics = {"test/content_loss": content_loss_value, "test/style_loss": style_loss_value, "test/fid_loss": fid_loss_value, "test/loss": loss, "test/metric_fid": metric_fid, "test/metric_ssim": ssim}

            reconstructed = sample_reconstructed.detach()

            del sample_images
            del target_images
            del sample_reconstructed
            del reconstructed_features
            del target_features

            torch.cuda.empty_cache()

        return reconstructed, metrics