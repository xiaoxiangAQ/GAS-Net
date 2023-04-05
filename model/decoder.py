import torch
import torch.nn as nn
import torch.nn.functional as F

class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class Decoder_GASN(nn.Module):
    def __init__(self, fc, BatchNorm):
        super(Decoder_GASN, self).__init__()
        self.fc = fc
        self.dr2 = DR(64, 64)
        self.dr3 = DR(128, 64)
        self.dr4 = DR(256, 64)
        self.dr5 = DR(512, 64)
        self.last_conv = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(128, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
                                       BatchNorm(self.fc),
                                       nn.ReLU(),
                                       )
        self._init_weight()
        self.FAM_feat  = [512, 512, 512, 256]
        self.FAM = FAM_layers(self.FAM_feat, in_channels=256, dilation=True)
        self.FAM_output_layer = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x, low_level_feat2, low_level_feat3, low_level_feat4):
        
        x2 = self.dr2(low_level_feat2)
        x3 = self.dr3(low_level_feat3)
        x4 = self.dr4(low_level_feat4)
        x = self.dr5(x)

        x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x2.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x, x2, x3, x4), dim=1)
        x = x + self.FAM(x)

        # x = torch.cat((self.FAM(x), x), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoderGASN(fc, BatchNorm):
    return Decoder_GASN(fc, BatchNorm)

def FAM_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 3
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)   