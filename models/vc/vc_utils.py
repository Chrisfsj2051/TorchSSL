import torch
import torch.nn.functional as F
from train_utils import ce_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


def consistency_loss(logits_s, logits_w, class_acc, p_target, p_model, name='ce',
                     T=1.0, p_cutoff=0.0, use_hard_labels=True, use_DA=False):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        if use_DA:
            if p_model == None:
                p_model = torch.mean(pseudo_label.detach(), dim=0)
            else:
                p_model = p_model * 0.999 + torch.mean(pseudo_label.detach(), dim=0) * 0.001
            pseudo_label = pseudo_label * p_target / p_model
            pseudo_label = (pseudo_label / pseudo_label.sum(dim=-1, keepdim=True))

        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()  # convex
        # for idx in range(10):
        #     print(idx, ' ', p_cutoff * (class_acc[idx] / (2. - class_acc[idx])))
        select = max_probs.ge(p_cutoff).long()
        if use_hard_labels:
             masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask, select, max_idx.long(), p_model

    else:
        assert Exception('Not Implemented consistency_loss')


def consistency_loss_prob(logits_s, logits_w, class_acc, p_target, p_model, name='ce',
                          T=1.0, p_cutoff=0.0, use_hard_labels=True, use_DA=False):
    assert name in ['ce']
    logits_w = logits_w.detach()
    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    # max_probs[max_probs>0.95] = 0
    select = torch.bernoulli(max_probs).long()
    mask = select.float()
    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
    else:
        pseudo_label = torch.softmax(logits_w / T, dim=-1)
        masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
    return masked_loss.mean(), mask, select, max_idx.long(), p_model
