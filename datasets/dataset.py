from torchvision import transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot, split_ssl_data
from .augmentation.randaugment import RandAugment

import torchvision
from PIL import Image
import numpy as np
import copy


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        if self.is_ulb:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3, 5))
        else:
            self.strong_transform = strong_transform

    def get_holdout_dset(self, args, holdout_num):
        assert holdout_num % self.num_classes == 0 and holdout_num < len(self.data)
        if holdout_num == 0:
            return self, None
        img_shape = self.data.shape[1:]
        data = self.data.reshape(self.num_classes, len(self.data) // self.num_classes, *img_shape)
        holdout_data = data[:, :holdout_num // self.num_classes].reshape(-1, *img_shape)
        lb_data = data[:, holdout_num // self.num_classes:].reshape(-1, *img_shape)
        targets = self.targets.reshape(self.num_classes, len(self.data) // self.num_classes)
        holdout_targets = targets[:, :holdout_num // self.num_classes].reshape(-1)
        lb_targets = targets[:, holdout_num // self.num_classes:].reshape(-1)
        lb_dset = BasicDataset(
            self.alg, lb_data, lb_targets,
            num_classes=self.num_classes,
            transform=self.transform,
            is_ulb=self.is_ulb,
            strong_transform=self.strong_transform,
            onehot=self.onehot)
        holdout_dset = BasicDataset(
            self.alg, holdout_data, holdout_targets,
            num_classes=self.num_classes,
            transform=self.transform,
            is_ulb=self.is_ulb,
            strong_transform=self.strong_transform,
            onehot=self.onehot)
        return lb_dset, holdout_dset

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """

        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images

        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.is_ulb:
                return idx, img_w, target
            else:
                if self.alg == 'fixmatch':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'vc':
                    # return target only for eval
                    return idx, img_w, self.strong_transform(img), target
                elif self.alg == 'flexmatch':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'pimodel':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'pseudolabel':
                    return idx, img_w
                elif self.alg == 'vat':
                    return idx, img_w
                elif self.alg == 'meanteacher':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'uda':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'mixmatch':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return idx, img_w, img_s1, img_s2, img_s1_rot, rotate_v_list.index(rotate_v1)
                elif self.alg == 'fullysupervised':
                    return idx

    def __len__(self):
        return len(self.data)
