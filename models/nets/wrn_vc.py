import torch

from models.nets.wrn import WideResNet
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WideResNetVariationCalibration(WideResNet):
    def __init__(self, num_classes, z_dim=64, **kwargs):
        super(WideResNetVariationCalibration, self).__init__(num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        self.z_dim = z_dim
        # p(r|c,z): self.decoder
        self.decoder = nn.Sequential(
            nn.Linear(num_classes + self.channels + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
            nn.Sigmoid()
        )
        # p(z|c,r)
        self.encoder = nn.Sequential(
            nn.Linear(num_classes + num_classes + self.channels, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * z_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def test_forward(self, x):
        return super(WideResNetVariationCalibration, self).forward(x)

    def calc_uncertainty(self, x):
        self.sampling_times = 20
        batch_size = x.shape[0]
        x = torch.cat([x for _ in range(self.sampling_times)], 0)
        with torch.no_grad():
            x = torch.dropout(x, p=0.5, train=True)
            pred = self.fc(x).argmax(1)
        pred_onehot = F.one_hot(pred, 10)
        pred_onehot = pred_onehot.reshape(batch_size, self.sampling_times, -1)
        pred_onehot = pred_onehot.sum(1).float() / self.sampling_times
        return pred_onehot

    def forward(self, x, ood_test=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        logits = self.fc(out)

        cali_gt_label = self.calc_uncertainty(out)
        encoder_x = torch.cat([logits, out, cali_gt_label], 1)
        h = self.encoder(encoder_x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterise(mu, logvar)
        recon_r = self.decoder(torch.cat([logits, out, z], 1))

        with torch.no_grad():
            h = torch.randn(x.shape[0], self.z_dim * 2).to(x.device)
            sample_mu, sample_logvar = h.chunk(2, dim=1)
            z = self.reparameterise(sample_mu, sample_logvar)
            decode_input = torch.cat([logits, out, z], 1)
            cali_output = self.decoder(decode_input)

        return logits, recon_r, cali_gt_label, (mu, logvar), cali_output


class build_WideResNetVariationCalibration:
    def __init__(self, first_stride=1, depth=28, widen_factor=2, bn_momentum=0.01, leaky_slope=0.0, dropRate=0.0,
                 use_embed=False, is_remix=False):
        self.first_stride = first_stride
        self.depth = depth
        self.widen_factor = widen_factor
        self.bn_momentum = bn_momentum
        self.dropRate = dropRate
        self.leaky_slope = leaky_slope
        self.use_embed = use_embed
        self.is_remix = is_remix

    def build(self, num_classes):
        return WideResNetVariationCalibration(
            first_stride=self.first_stride,
            depth=self.depth,
            num_classes=num_classes,
            widen_factor=self.widen_factor,
            drop_rate=self.dropRate,
            is_remix=self.is_remix,
        )


if __name__ == '__main__':
    wrn_builder = build_WideResNetVariationCalibration(1, 10, 2, 0.01, 0.1, 0.5)
    wrn = wrn_builder.build(10)
    print(wrn)