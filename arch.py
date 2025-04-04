import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class Decoder(nn.Module):
    def __init__(self, latent_dim=768, output_size=128):
        super(Decoder, self).__init__()
        
        self.initial_size = output_size // 16  # 8*8
        
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.initial_size * self.initial_size),
            nn.LeakyReLU(0.2)
        )
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            # 512 x 8 x 8 -> 256 x 16 x 16
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                ResBlock(256, 256)
            ),
            # 256 x 16 x 16 -> 128 x 32 x 32
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                ResBlock(128, 128)
            ),
            # 128 x 32 x 32 -> 64 x 64 x 64
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                ResBlock(64, 64)
            ),
            # 64 x 64 x 64 -> 32 x 128 x 128
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2)
            )
        ])
        
        # Final layer to get 3 channels
        self.to_rgb = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.projection(x)
        x = x.view(-1, 512, self.initial_size, self.initial_size)
        
        for block in self.up_blocks:
            x = block(x)
        
        x = self.to_rgb(x)
        
        return x

if __name__ == "__main__":
    latent_vector = torch.randn(1, 768)
    decoder = Decoder()
    output_image = decoder(latent_vector)
    print("Output shape:", output_image.shape)
    
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters: {total_params:,}")
