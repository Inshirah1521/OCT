import torch
import torch.nn as nn

################################################################
class multi_unet_model1(nn.Module):
    def __init__(self, n_classes=7, IMG_HEIGHT=512, IMG_WIDTH=240, IMG_CHANNELS=1):
        super(multi_unet_model1, self).__init__()
        
        # Contraction path
        self.c1_1 = nn.Conv2d(IMG_CHANNELS, 16, kernel_size=3, padding=1)
        self.c1_drop = nn.Dropout2d(0.1)
        self.c1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.p1 = nn.MaxPool2d((2, 2))
        
        self.c2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.c2_drop = nn.Dropout2d(0.1)
        self.c2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool2d((2, 2))
         
        self.c3_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.c3_drop = nn.Dropout2d(0.2)
        self.c3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.p3 = nn.MaxPool2d((2, 2))
         
        self.c4_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.c4_drop = nn.Dropout2d(0.2)
        self.c4_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.p4 = nn.MaxPool2d((2, 2))
         
        self.c5_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.c5_drop = nn.Dropout2d(0.3)
        self.c5_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Expansive path 
        self.u6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c6_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.c6_drop = nn.Dropout2d(0.2)
        self.c6_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
         
        self.u7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c7_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.c7_drop = nn.Dropout2d(0.2)
        self.c7_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
         
        self.u8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c8_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.c8_drop = nn.Dropout2d(0.1)
        self.c8_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
         
        self.u9 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.c9_1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.c9_drop = nn.Dropout2d(0.1)
        self.c9_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
         
        self.outputs = nn.Conv2d(16, n_classes, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, inputs):
        # Contraction path
        c1 = torch.relu(self.c1_1(inputs))
        c1 = self.c1_drop(c1)
        c1 = torch.relu(self.c1_2(c1))
        p1 = self.p1(c1)
        
        c2 = torch.relu(self.c2_1(p1))
        c2 = self.c2_drop(c2)
        c2 = torch.relu(self.c2_2(c2))
        p2 = self.p2(c2)
         
        c3 = torch.relu(self.c3_1(p2))
        c3 = self.c3_drop(c3)
        c3 = torch.relu(self.c3_2(c3))
        p3 = self.p3(c3)
         
        c4 = torch.relu(self.c4_1(p3))
        c4 = self.c4_drop(c4)
        c4 = torch.relu(self.c4_2(c4))
        p4 = self.p4(c4)
         
        c5 = torch.relu(self.c5_1(p4))
        c5 = self.c5_drop(c5)
        c5 = torch.relu(self.c5_2(c5))
        
        # Expansive path 
        u6 = self.u6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = torch.relu(self.c6_1(u6))
        c6 = self.c6_drop(c6)
        c6 = torch.relu(self.c6_2(c6))
         
        u7 = self.u7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = torch.relu(self.c7_1(u7))
        c7 = self.c7_drop(c7)
        c7 = torch.relu(self.c7_2(c7))
         
        u8 = self.u8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = torch.relu(self.c8_1(u8))
        c8 = self.c8_drop(c8)
        c8 = torch.relu(self.c8_2(c8))
         
        u9 = self.u9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = torch.relu(self.c9_1(u9))
        c9 = self.c9_drop(c9)
        c9 = torch.relu(self.c9_2(c9))
         
        outputs = self.outputs(c9)
         
        return outputs

    def _initialize_weights(self):
        """Initialize weights using He normal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)