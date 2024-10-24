import torch
import torch.nn as nn
from torch.nn.functional import relu


class UNet3D(nn.Module):
    def __init__(self, n_class, use_batchnorm=False, binary=False):
        super(UNet3D, self).__init__()
        
        self.use_batchnorm = use_batchnorm

        # If True, this will be binary segmentation (sigmoid), else multi-class (softmax)
        self.binary = binary
        
        # Encoder
        # Input channels changed from 2D to 3D
        self.e11 = nn.Conv3d(8, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.e21 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.e31 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.e41 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.e51 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv3d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv3d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv3d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv3d(64, n_class, kernel_size=1)

        # Initialize weights
        self.init_weights()


    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        #xu1 = self.upconv1(xe52)  # Upsample
        print(f"Shape of xu1 (after upconv1): {xu1.shape}")  # Print shape for debugging
        print(f"Shape of xe42 (encoder skip connection): {xe42.shape}")  # Print shape for debugging

        # Pad the encoder output (xe42) to match the depth of xu1
        if xu1.shape[2] != xe42.shape[2]:  # Check if depth doesn't match
            xe42 = F.pad(xe42, (0, 0, 0, 0, 0, xu1.shape[2] - xe42.shape[2]))  # Pad depth dimension
    
        xu11 = torch.cat([xu1, xe42], dim=1)  # Concatenate skip connection from encoder
        print(f"Shape of concatenated tensor xu11: {xu11.shape}")  # Should be [batch_size, 1024, depth, height, width]

        #xu11 = torch.cat([xu1, xe42], dim=1)  # Concatenates 512 from upconv1 and 512 from xe42
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)  # Concatenates 256 from upconv2 and 256 from xe32
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)  # Concatenates 128 from upconv3 and 128 from xe22
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)  # Concatenates 64 from upconv4 and 64 from xe12
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        # Apply final activation
        if self.binary:
            out = torch.sigmoid(out)  # Binary segmentation
        else:
            out = torch.softmax(out, dim=1)  # Multi-class segmentation

        return out


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
