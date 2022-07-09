import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import wide_resnet50_2
from torchvision.models._utils import IntermediateLayerGetter

abs_file = os.path.abspath(__file__)
abs_dir = abs_file[:abs_file.rfind('\\')] if os.name == 'nt' else abs_file[:abs_file.rfind(r'/')]


def embedding_concat(f_1, f_2):
    B, C1, H1, W1 = f_1.size()
    _, C2, H2, W2 = f_2.size()

    scale = int(H1 / H2)
    f_1 = F.unfold(f_1, kernel_size=scale, dilation=1, stride=scale)
    f_1 = f_1.view(B, C1, -1, H2, W2)
    
    f_2 = f_2.unsqueeze(2).expand(-1, -1, f_1.size(2), -1, -1)
    f_out = torch.cat([f_1, f_2], dim=1)

    f_out = f_out.view(B, -1, H2*W2)
    f_out = F.fold(f_out, kernel_size=scale, output_size=(H1, W1), stride=scale)
    return f_out


class ProtoAD(nn.Module):
    """ ProtoAD """
    def __init__(self):
        super(ProtoAD, self).__init__()
        model = wide_resnet50_2(pretrained=True)
        return_layers = {'layer1': 'f_1', 'layer2': 'f_2', 'layer3': 'f_3'}
        self.backbone = IntermediateLayerGetter(model, return_layers)

    def forward(self, x):
        ''' forward pass '''
        features = self.backbone(x)
        embedding = embedding_concat(features['f_2'], features['f_3'])  # layer2 + layer3
        embedding = embedding_concat(features['f_1'], embedding) # layer12 + layer3     # [B, C, h, w]
        return embedding

    def update_prototype(self, cluster_center):
        """ cluster_center: dim K x C """
        self.prototpe = nn.Parameter(cluster_center).unsqueeze(-1).unsqueeze(-1)  # (K, C, 1, 1)

    def anomaly_forward(self, x):
        ''' forward pass for anomaly detection and localization '''
        embedding = self.forward(x)
        out = F.normalize(embedding, p=2, dim=1)
        cosine_sim = F.conv2d(out, self.prototpe, bias=None, stride=1, padding=0)   # [B, K, h, w]
        anomaly_map = 1 - cosine_sim.max(1)[0]  # [B, h, w]
        return anomaly_map
