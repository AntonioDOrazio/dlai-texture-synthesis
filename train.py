import torch
from tqdm import tqdm
from model import TextureSynthesizer
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import os 
from PIL import Image
import random
import wandb
wandb.init(project="dlai-texture-synthesis")

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 600
BATCH_SIZE = 24 
LEARNING_RATE =  0.0032 
SAMPLE_DIR = "./dataset/textures-label/x"
GROUND_TRUTH_DIR = "./dataset/textures-label/y"
TRAIN_PERCENTAGE = 20
TEST_PERCENTAGE = 5
STYLE_LOSS ="wasserstein"

SAVE_DIR = "train_results"
os.makedirs(SAVE_DIR, exist_ok=True)

wandb.config = {"learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE, "style_loss": STYLE_LOSS}

# Scheduler settings
STEP_SIZE=150
GAMMA = 0.1
SEED = 128
random.seed(SEED)

# Define the transformations to be applied to the images
x_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])

y_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor()
])



# Create a custom dataset that loads both sample and ground truth images
class PairedImageDataset(Dataset):
    def __init__(self, sample_dir, ground_truth_dir, percentage, x_transform=None, filter_indices=None, y_transform=None):

        self.sample_files = self.get_image_files(sample_dir)
        self.ground_truth_files = self.get_image_files(ground_truth_dir)

        if percentage <= 0 or percentage > 100:
            raise ValueError("Percentages should be between 0 and 100")

        num_samples = int(len(self.sample_files) * (percentage / 100))

        if filter_indices is not None:
            samples = set(range(len(self.sample_files))) - set(filter_indices)
            indices = random.sample(list(samples), num_samples)
        else:
            indices = random.sample(range(len(self.sample_files)), num_samples)

        self.indices = indices

        self.sample_files = [self.sample_files[i] for i in indices]
        self.ground_truth_files = [self.ground_truth_files[i] for i in indices]

        self.x_transform = x_transform
        self.y_transform = y_transform

    def __getitem__(self, index):

        sample_file = self.sample_files[index]
        ground_truth_file = self.ground_truth_files[index]

        sample_image = self.load_image(sample_file)
        ground_truth_image = self.load_image(ground_truth_file)

        if self.x_transform is not None:
            sample_image = self.x_transform(sample_image)
        if self.y_transform is not None:
            ground_truth_image = self.y_transform(ground_truth_image)

        return sample_image, ground_truth_image

    def __len__(self):
        return len(self.sample_files)

    def get_image_files(self, directory):
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if self.is_image_file(file):
                    image_files.append(os.path.join(root, file))
        return image_files

    def is_image_file(self, filename):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        return any(filename.lower().endswith(ext) for ext in image_extensions)

    def load_image(self, file):
        with Image.open(file) as img:
            return img.convert("RGB")

train_dataset = PairedImageDataset(GROUND_TRUTH_DIR, GROUND_TRUTH_DIR, TRAIN_PERCENTAGE, x_transform=x_transform, y_transform=y_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = PairedImageDataset(GROUND_TRUTH_DIR, GROUND_TRUTH_DIR, TEST_PERCENTAGE, x_transform=x_transform, y_transform=y_transform, filter_indices=train_dataset.indices)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = TextureSynthesizer(64, 64, train=True, encoder_type="vgg").to(DEVICE)
optimizer = model.configure_optimizers(lr=LEARNING_RATE, step_size=STEP_SIZE, gamma=GAMMA)

wandb.watch(model)

import numpy as np

global_steps_train = 0
global_steps_test  = 0


wandb.define_metric("train/step")
wandb.define_metric("train/*", step_metric="train/step")
wandb.define_metric("test/step")
wandb.define_metric("test/*", step_metric="test/step")
wandb.define_metric("epoch")
wandb.define_metric("avg_train_loss_epoch", step_metric="epoch")
wandb.define_metric("avg_test_loss_epoch", step_metric="epoch")



for epoch in range(NUM_EPOCHS):
    avg_train_loss_epoch = []
    avg_test_loss_epoch  = []

    '''
        # TRAIN LOOP
    '''
    loop = tqdm(enumerate(train_loader))
    

    for i, (sample_images, target_images) in loop:

        sample_reconstructed, metrics = model.training_step((sample_images, target_images), i, style_loss_type=STYLE_LOSS)
        
        avg_train_loss_epoch.append(metrics["train/loss"].item())
        loop.set_postfix(loss=metrics["train/loss"].item(), content_loss=metrics["train/content_loss"].item(), style_loss=metrics["train/style_loss"].item(), fid_loss=metrics["train/fid_loss"].item())
        
        metrics["train/step"] = global_steps_train
        wandb.log(metrics)

        global_steps_train+=1

    wandb.log({"avg_train_loss_epoch": np.mean(avg_train_loss_epoch), "epoch": epoch})
    model.scheduler.step()


    with torch.no_grad():
        
        #random_indices = random.sample(range(len(sample_reconstructed)), k=min(15, len(sample_reconstructed)))
        epoch_dir = os.path.join(SAVE_DIR, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        for idx in range(len(sample_reconstructed)):

            image = sample_reconstructed[idx].cpu()
            save_image(image, os.path.join(epoch_dir, f"image_{idx}.png"))

            image = target_images[idx].cpu()
            save_image(image, os.path.join(epoch_dir, f"image_{idx}_y.png"))

        # Create a new tensor to hold the concatenated images
        concatenated_img = torch.cat((target_images, sample_reconstructed.cpu()), dim=2)

        image = wandb.Image(
            concatenated_img, 
            caption="Epoch {}. Top: Input, Bottom: Texture".format(epoch+1)
        )
        wandb.log({"results": image})

    '''
    if epoch % 50 == 0:
        architecture_path = os.path.join(SAVE_DIR, f"model_architecture_{epoch}.pth")
        torch.save(model.state_dict(), architecture_path)
        # Save model weights
        weights_path = os.path.join(SAVE_DIR, f"model_weights_{epoch}.pth")
        torch.save(model.state_dict(), weights_path)
    '''
    if epoch+1 == NUM_EPOCHS:
        weights_path = os.path.join(SAVE_DIR, f"model_weights_{epoch}.pth")
        torch.save(model.state_dict(), weights_path)
        
    '''
        # TEST LOOP
    '''
    loop = tqdm(enumerate(test_loader))
 
    for i, (sample_images, target_images) in loop:

        sample_reconstructed, metrics = model.validation_step((sample_images, target_images), i, style_loss_type=STYLE_LOSS)

        avg_test_loss_epoch.append(metrics["test/loss"].item())
        loop.set_postfix(loss=metrics["test/loss"].item(), content_loss=metrics["test/content_loss"].item(), style_loss=metrics["test/style_loss"].item(), fid_loss=metrics["test/fid_loss"].item())
        
        metrics["test/step"] = global_steps_test
        wandb.log(metrics)

        global_steps_test+=1


    wandb.log({"avg_test_loss_epoch": np.mean(avg_test_loss_epoch), "epoch": epoch})

    concatenated_img = torch.cat((target_images, sample_reconstructed.cpu()), dim=2)
    image = wandb.Image(
        concatenated_img, 
        caption="Epoch {}. Top: Input, Bottom: Texture".format(epoch+1)
    )
    wandb.log({"test_results": image})

    del sample_reconstructed

    with torch.no_grad():
        torch.cuda.empty_cache()


print("TRAINING COMPLETE!")

artifact = wandb.Artifact('model', type='model')
artifact.add_file(weights_path)
wandb.run.log_artifact(artifact)
wandb.run.finish()

print(f"Model architecture and weights saved at {SAVE_DIR}")

