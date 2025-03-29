import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim=768):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 512x8x8 -> 256x16x16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 256x16x16 -> 128x32x32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 128x32x32 -> 64x64x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), # 64x64x64 -> 3x128x128
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 8, 8)
        x = self.deconv_layers(x)
        return x

if __name__ == "__main__":
    latent_vector = torch.randn(1, 768)
    decoder = Decoder()
    output_image = decoder(latent_vector)
    print("Output shape:", output_image.shape)
