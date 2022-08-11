import torch

from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class IPDF(nn.Module):
    def __init__(self):
        super().__init__()
        # See: https://github.com/google-research/google-research/tree/master/implicit_pdf#reproducing-symsol-results
        # and Section S8.
        self.cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        visual_embedding_size = self.cnn.layer4[2].bn3.num_features
        self.L = 3
        R_feats = 2 * self.L * 9
        n_hidden_nodes = 256
        self.mlp = nn.Sequential(
            nn.Linear(visual_embedding_size + R_feats, n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(n_hidden_nodes, n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(n_hidden_nodes, n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(n_hidden_nodes, n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(n_hidden_nodes, 1),
        )

    def get_probs(self, imgs, Rs_fake_Rs):
        x = self.cnn.conv1(imgs)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)

        x = self.cnn.avgpool(x)
        x = torch.flatten(x, 1).unsqueeze(1).repeat(1, Rs_fake_Rs.shape[1], 1)

        Rs_fake_Rs_encoded = []
        for l_pos in range(self.L):
            Rs_fake_Rs_encoded.append(torch.sin(2**l_pos * torch.pi * Rs_fake_Rs))
            Rs_fake_Rs_encoded.append(torch.cos(2**l_pos * torch.pi * Rs_fake_Rs))

        Rs_fake_Rs_encoded = torch.cat(Rs_fake_Rs_encoded, dim=-1)
        x = torch.cat([x, Rs_fake_Rs_encoded], dim=-1)
        x = self.mlp(x).squeeze(2)
        probs = torch.softmax(x, 1)
        return probs

    def forward(self, imgs, Rs_fake_Rs):
        # See: https://github.com/google-research/google-research/blob/207f63767d55f8e1c2bdeb5907723e5412a231e1/implicit_pdf/models.py#L188
        # and Equation (2) in the paper.
        V = torch.pi**2 / Rs_fake_Rs.shape[1]
        probs = 1 / V * self.get_probs(imgs, Rs_fake_Rs)[:, 0]
        return probs
