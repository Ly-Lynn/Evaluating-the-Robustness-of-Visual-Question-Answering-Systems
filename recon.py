import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from arch import Decoder
from utils import resize_image, convert_torch, convert_numpy
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import random
from metrics import L2
from torchvision import models
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def create_window(self, window_size, channel):
        def _gaussian(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) for x in range(window_size)])
            return gauss / gauss.sum()
            
        _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        if img1.shape != img2.shape:
            print(f"Shape mismatch: {img1.shape} vs {img2.shape}")
            if img1.dim() == 3 and img2.dim() == 4:
                img1 = img1.unsqueeze(0)
            elif img1.dim() == 4 and img2.dim() == 3:
                img2 = img2.unsqueeze(0)            
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
            
        device = img1.device
        window = self.window.to(device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIM(window_size=11, size_average=True, channel=3)
        self.alpha = alpha  # MSE weight
        self.beta = beta  # SSIM weight

    def forward(self, output, target):
        mse = self.mse_loss(output, target)
        ssim_value = 1 - self.ssim_loss(output, target)  # 1 - SSIM for loss
        
        total_loss = self.alpha * mse  + self.beta * ssim_value
        return total_loss

class ReconstructorTrainer:
    def __init__(self, 
                decoder,
                optimizer,
                scheduler,
                criterion=None,
                device='cuda',
                output_dir='result',
                use_augmentation=True,
                regularization_weight=1e-5):
        self.device = device
        self.decoder = decoder
        self.use_multi_gpu = False
        if torch.cuda.device_count() > 1:
            self.use_multi_gpu = True
            self.decoder = DataParallel(self.decoder)
            self.decoder.to(self.device)

        self.optimizer = optimizer
        self.criterion = CombinedLoss() if criterion is None else criterion
        self.scheduler = scheduler
        self.logger = []
        self.output = output_dir
        self.use_augmentation = use_augmentation
        self.regularization_weight = regularization_weight
        os.makedirs(self.output, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.best_epoch = 0

    def get_logger(self):
        return self.logger
        
    def save_log_loss(self, loss, epoch):
        self.logger.append(f"{loss}")
        with open(os.path.join(self.output, 'loss.txt'), 'a') as f:
            f.write(f"{epoch} {loss:.6f}\n")
            
    def save_model(self, path):
        path = os.path.join(self.output, path)
        model_to_save = self.decoder.module if self.use_multi_gpu else self.decoder
        
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        
    def load_model(self, path):
        checkpoint = torch.load(path)
        if self.use_multi_gpu:
            self.decoder.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.decoder.load_state_dict(checkpoint['model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
    def save_image(self, image, path):
        path = os.path.join(self.output, path)
        img = image.cpu().detach().numpy()[0]  
        if img.shape[0] == 3:  
            img = np.transpose(img, (1, 2, 0)) 
        
        img = np.clip(img, 0, 1)  
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        pil_img.save(path)
    
    def augment_latent(self, latent_vectors, strength=0.05):
        """Apply random noise to latent vectors for augmentation"""
        if not self.use_augmentation:
            return latent_vectors
            
        noise = torch.randn_like(latent_vectors) * strength
        return latent_vectors + noise
    
    def train_step(self, latent_vectors, target_images):
        self.decoder.train()
        latent_vectors = latent_vectors.to(self.device)
        target_images = target_images.to(self.device)
        
        augmented_latent = self.augment_latent(latent_vectors)
        
        self.optimizer.zero_grad()
        output_images = self.decoder(augmented_latent)
        
        reconstruction_loss = self.criterion(output_images, target_images)
        
        l2_reg = 0
        for param in self.decoder.parameters():
            l2_reg += torch.norm(param)
        
        total_loss = reconstruction_loss + self.regularization_weight * l2_reg
        
        total_loss.backward()
        self.optimizer.step()
        
        return output_images, reconstruction_loss.item()
    
    def validate(self, latent_vectors, target_images):
        """Perform validation on non-augmented data"""
        self.decoder.eval()
        latent_vectors = latent_vectors.to(self.device)
        target_images = target_images.to(self.device)
        
        with torch.no_grad():
            output_images = self.decoder(latent_vectors)
            loss = self.criterion(output_images, target_images)
            
        return output_images, loss.item()
    
    def train(self, train_latent_vectors, train_target_images, 
              val_latent_vectors=None, val_target_images=None, 
              epochs=1000, patience=50, save_interval=100):
        """
        Train the reconstructor with early stopping based on validation loss
        """
        best_loss = float('inf')
        patience_counter = 0
        
        if val_latent_vectors is None or val_target_images is None:
            split = int(0.9 * len(train_latent_vectors))
            val_latent_vectors = train_latent_vectors[split:]
            val_target_images = train_target_images[split:]
            train_latent_vectors = train_latent_vectors[:split]
            train_target_images = train_target_images[:split]
        
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            # Training step
            img, train_loss = self.train_step(train_latent_vectors, train_target_images)
            self.train_losses.append(train_loss)
            
            # Validation step
            val_img, val_loss = self.validate(val_latent_vectors, val_target_images)
            self.val_losses.append(val_loss)
            
            # Log losses
            self.save_log_loss(val_loss, epoch)
            
            if val_loss < best_loss:
                best_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0
                # self.save_image(val_img, f"best_image_epoch_{epoch}.png")
                self.save_model(f"best_model.pth")
            else:
                patience_counter += 1
            
            if (epoch + 1) % save_interval == 0:
                self.save_model(f"checkpoint_epoch_{epoch}.pth")
                self.save_image(img, f"train_image_epoch_{epoch}.png")
            
            if (epoch + 1) % (epochs // 20) == 0:
                print(f"\nEpoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best epoch: {self.best_epoch}")
                break
                
            self.scheduler.step()
        
        self.load_model(os.path.join(self.output, "best_model.pth"))
        return best_loss
    
    def eval(self, org_img, latent_vectors, save_path=None):
        self.decoder.eval()
        latent_vectors = latent_vectors.to(self.device)
        org_img = org_img.to(self.device)
        
        with torch.no_grad():
            output_images = self.decoder(latent_vectors)
        
        # Calculate metrics
        mse_loss = F.mse_loss(output_images, org_img).item()
        
        ssim_module = SSIM(window_size=11, size_average=True, channel=output_images.shape[1])
        ssim_value = ssim_module(output_images, org_img).item()
        
        if save_path:
            self.save_image(output_images, save_path)
        
        return output_images, {'mse': mse_loss, 'ssim': ssim_value}
    
    def reconstruct_batch(self, latent_vectors, batch_size=16):
        self.decoder.eval()
        all_outputs = []
        
        num_vectors = latent_vectors.size(0)
        
        for i in range(0, num_vectors, batch_size):
            end = min(i + batch_size, num_vectors)
            batch = latent_vectors[i:end].to(self.device)
            
            with torch.no_grad():
                outputs = self.decoder(batch)
                all_outputs.append(outputs.cpu())
        
        return torch.cat(all_outputs, dim=0)

if __name__ == "__main__":
    batch_size = 100
    # img_path = r'D:\codePJ\RESEARCH\Evaluating-the-Robustness-of-Visual-Question-Answering-Systems\test\dog1.jpg'
    img_path = r'/kaggle/working/Evaluating-the-Robustness-of-Visual-Question-Answering-Systems/test/dog1.jpg'
    image = Image.open(img_path)
    image = resize_image(image, (128, 128)).cuda()
    # Convert the tensor image to a PIL image and save it
    img = image.cpu().detach().numpy()[0]  # remove batch dimension if present
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    pil_img.save("org_image.png")


    target_image = image.unsqueeze(0)  
    target_images = image.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    print("Image shape:", target_images.shape)


    # ------------------------- TEST PARAMS -------------------------
    decoder = Decoder(latent_dim=768)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    reconstructor = ReconstructorTrainer(decoder=decoder, 
                                         optimizer=optimizer, scheduler=scheduler)
    

    # ------------------------- TEST TRAIN -------------------------
    latent_vectors = torch.randn(batch_size, 768)
    train_vectors = latent_vectors[:int(0.8 * batch_size)]
    val_vectors = latent_vectors[int(0.8 * batch_size):]
    train_target_images = target_images[:int(0.8 * batch_size)]
    val_target_images = target_images[int(0.8 * batch_size):]
    print("Train shape:", train_vectors.shape, train_target_images.shape)
    print("Val shape:", val_vectors.shape, val_target_images.shape)
    
    trainer = ReconstructorTrainer(
        decoder=decoder,
        optimizer=optimizer,
        scheduler=scheduler,
        device='cuda',
        output_dir='improved_results',
        use_augmentation=True
    )

    trainer.train(
        train_target_images=train_target_images,
        train_latent_vectors=train_vectors,
        val_target_images=val_target_images,
        val_latent_vectors=val_vectors,
        epochs=500,
        patience=50,
    )
    
    # ------------------------- TEST INFER ON 1 IMG -------------------------
    model = reconstructor.load_model(r"D:\codePJ\RESEARCH\Evaluating-the-Robustness-of-Visual-Question-Answering-Systems\improved_results\best_model.pth")
    test_inference_vec = torch.randn(1, 768)
    test_inf, l2 = model.eval(image.unsqueeze(0), test_inference_vec)
    print("output ", test_inf.shape)

    print("L2 distance:", l2)
    reconstructor.save_image(test_inf, "test_inf.png")
    # reconstructor.save_model("test_model.pth")
