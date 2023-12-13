import torch.nn as nn
from torch import cat
import modules.activation_funcs as activation_funcs



class BatchDoubleConv(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 activation_function):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1,
                      bias=False),

            nn.BatchNorm2d(out_channels),

            activation_function,

            nn.Conv2d(in_channels=out_channels, 
                      out_channels=out_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1,
                      bias=False),
            
            nn.BatchNorm2d(out_channels),

            activation_function
        )

    def forward(self, x):
        return self.conv(x)


class BatchSegUnet(nn.Module):
    def __init__(self, options):
        super().__init__()

        self.in_channels = options["in_channels"]
        self.out_channels = options["out_channels"]
        self.activation_function = activation_funcs.get_activation_function(options["activation_function"])
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features=[64, 128, 256, 512, 1024]

        # Down part
        for feature in self.features:
            self.downs.append(BatchDoubleConv(self.in_channels, feature, self.activation_function))
            self.in_channels = feature

        # Up part    
        for feature in reversed(self.features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(BatchDoubleConv(feature*2, feature, self.activation_function))

        # Bottleneck   
        self.bottleneck = BatchDoubleConv(self.features[-1], 2*self.features[-1], self.activation_function)

        # Final
        self.final_conv = BatchDoubleConv(self.features[0], self.out_channels, self.activation_function)

    def forward(self, x):
        skip_connections = []

        # Down
        # 3x1024x1024 -> 64x512x512 -> 128x256x256 -> 256x128x128 -> 512x64x64 -> 1024x32x32
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        # 1024x32x32 -> 2048x32x32
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Up
        # 2048x32x32 -> 1024x64x64 -> 512x128x128 -> 256x128x128 -> 128x256x256 -> 64x512x512 
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            x = cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
        
        # Final
        # 64x1024x1024 -> 1x1024x1024
        return nn.functional.sigmoid(self.final_conv(x))


