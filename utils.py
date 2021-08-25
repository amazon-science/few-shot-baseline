# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from copy import deepcopy
import matplotlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
from tqdm import tqdm

from fastai.callbacks import *
from fastai.distributed import *
from fastai.script import *
from fastai.vision import *
from fastai.vision.models.wrn import WideResNet as wrn

# resnet152 with adaptive pool at the end of the feature extractor
from torchvision.models import resnet152

torch.backends.cudnn.benchmark = True

def set_seed(seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_params(dataset, architecture, train_and_val):

    params = {}

    if dataset in ['Mini-ImageNet', 'Tiered-ImageNet']:
        params['image_size'] = 84
    elif dataset in ['CIFAR-FS', 'FC-100']:
        params['image_size'] = 32

    if train_and_val:
        params['csv_file_name'] = 'train_and_val.csv'
        params['parameter_file_name'] = dataset + '_' + architecture + '_TV'
    else:
        params['csv_file_name'] = 'train.csv'
        params['parameter_file_name'] = dataset + '_' + architecture
    params['parameter_path'] = os.path.join(dataset, 'models',
        params['parameter_file_name'] + '.pth')

    params['cycle_length'] = 40
    params['relu'] = True
    params['lambda'] = 1.

    return params

def get_transformations(image_size):

    transformations = [
        flip_lr(p=0.5),
        *rand_pad(padding=4, size=image_size, mode='reflection'),
        brightness(change=(0.1, 0.9)),
        contrast(scale=(0.6, 1.4))
        ]
    return transformations

def get_data(dataset, train_file_name, validation_file_name, image_size,
    batch_size):

    train_list = ImageList.from_csv(dataset, train_file_name)
    validation_list = ImageList.from_csv(dataset, validation_file_name)
    loaders = ItemLists(dataset, train_list, validation_list) \
        .label_from_df() \
        .transform((get_transformations(image_size), []), size=image_size) \
        .databunch(bs=batch_size, num_workers=4) \
        .normalize(imagenet_stats)
    return loaders

class FewShotDataset(torch.utils.data.Dataset):

    def save_images(self, dataset, file_name):

        data = pd.read_csv(os.path.join(dataset, file_name))
        classes = np.unique(data['c']).tolist()
        self.images = {
            cls : data.loc[data['c'] == cls]['fn'].tolist() for cls in classes
            }

    def __init__(self, dataset, file_name, image_size, way, support_shot,
        query_shot):

        self.dataset = dataset
        self.save_images(dataset, file_name)
        self.image_size = image_size
        self.way = way
        self.support_shot = support_shot
        self.query_shot = query_shot

    def get_way(self):

        return self.way

    def get_query_shot(self, classes):

        query_shot = {cls : self.query_shot for cls in classes}
        return query_shot

    def get_support_shot(self, classes, query_shot):

        support_shot = {cls : self.support_shot for cls in classes}
        return support_shot

    def __getitem__(self, idx):

        found_episode = False
        while not found_episode:
            found_episode = True
            way = self.get_way()
            classes = np.random.choice(list(self.images.keys()), way,
                replace=False)
            classes = sorted(classes)
            query_shot = self.get_query_shot(classes)
            support_shot = self.get_support_shot(classes, query_shot)
            support = dict(images=[], classes=[])
            query = dict(images=[], classes=[])
            for cls in classes:
                try:
                    images = np.random.choice(self.images[cls],
                        support_shot[cls] + query_shot[cls], replace=False)
                except:
                    found_episode = False
                    break
                support['images'] += images[: support_shot[cls]].tolist()
                support['classes'] += ([cls] * support_shot[cls])
                query['images'] += images[support_shot[cls] :].tolist()
                query['classes'] += ([cls] * query_shot[cls])

        support = pd.DataFrame(
            {'fn' : support['images'], 'c' : support['classes']}
            )
        query = pd.DataFrame(
            {'fn' : query['images'], 'c' : query['classes']}
            )

        support = ImageList.from_df(support, self.dataset).split_none() \
            .label_from_df().train
        query = ImageList.from_df(query, self.dataset).split_none() \
            .label_from_df().train
        for ind in range(len(query.y.items)):
            query.y.items[ind] = \
                support.y.classes.index(query.y.classes[query.y.items[ind]])
        query.y.classes = support.y.classes

        support = ItemLists(self.dataset, support, support) \
            .transform((get_transformations(self.image_size), []),
                size=self.image_size) \
            .databunch(bs=len(support), num_workers=0) \
            .normalize(imagenet_stats)
        query = ItemLists(self.dataset, query, query) \
            .transform((get_transformations(self.image_size), []),
                size=self.image_size) \
            .databunch(bs=len(query), num_workers=0) \
            .normalize(imagenet_stats)

        return support, query

def micro_forward(model, x, y, loss_func=None, loss_coef=None):

    num = x.size(0)
    yhs = []
    fs = []
    model.zero_grad()
    for x, y in zip(torch.split(x, 75), torch.split(y, 75)):
        yh = model(x)
        yhs.append(yh)
        if loss_func:
            f = x.size(0) * loss_func(yh, y) / num
            (loss_coef * f).backward()
            fs.append(f)
    yh = torch.cat(yhs)
    if loss_func:
        f = torch.stack(fs).sum()
        return yh, f
    else:
        return yh

def get_classifier(yh, y):

    classifier = torch.zeros((y.unique().size(0), yh.size(1))).cuda()
    for cls in torch.sort(y.unique())[0]:
        classifier[cls] = yh[y == cls].mean(dim=0)
    classifier = torch.nn.functional.normalize(classifier)
    return classifier

class Hardness:

    '''
    Hardness Metric
    Intuitively, classification performance on a few-shot episode is determined
    by the relative location of the features corresponding to labeled and
    unlabeled samples. If the unlabeled features are close to the labeled
    features from the same class, a classifier can distinguish between the
    classes easily to obtain a high accuracy. Otherwise, the accuracy would be
    low. We define hardness as the average log-odds of a test datum being
    classified incorrectly. We use the features of a generic feature extractor
    (ResNet-152, pre-trained on ImageNet) to calculate this metric. The labeled
    samples form class-specific cluster centers. The cluster affinities are
    calculated using cosine-similarities, followed by the softmax operator to
    get the probability distribution over the classes.
    '''

    def __init__(self):

        self.model = resnet152(pretrained=True)
        self.model = self.model.cuda()
        self.model.eval()

    def get_hardness(self, support, query):

        with torch.no_grad():
            for xs, ys in support.valid_dl:
                break
            yhs = micro_forward(self.model, xs, ys)
            classifier = get_classifier(yhs, ys)
            for xq, yq in query.valid_dl:
                break
            yhq = micro_forward(self.model, xq, yq)
            yhq = torch.nn.functional.normalize(yhq)
            yhq = yhq @ classifier.t()
            p = torch.softmax(yhq, dim=1)
            p = p[torch.arange(0, yq.size(0)), yq]
            hardness = ((1. - p) / p).log().mean().item()
        return hardness

class FewShotModel(torch.nn.Module):

    def __init__(self, backbone, support, relu):

        super().__init__()

        '''
        Support-Based Initialization
        Given the pre-trained model (backbone), we append a ReLU layer, an
        l2-normalization layer and a fully-connected layer that takes the
        logits of the backbone as input and predicts the few-shot labels. We
        calculate the per-class l2-normalized average features to initialize
        the weights of the fully-connected layer, with the biases set to 0.
        '''

        self.backbone = deepcopy(backbone)
        self.backbone.eval()

        self.relu = relu

        with torch.no_grad():
            for x, y in support.valid_dl:
                break
            yh = micro_forward(self.backbone, x, y)
            if self.relu:
                yh = torch.relu(yh)
            classifier = get_classifier(yh, y)

            self.classifier = torch.nn.Linear(classifier.size(1),
                classifier.size(0))
            self.classifier = self.classifier.cuda()
            self.classifier.weight.data.copy_(classifier)
            self.classifier.bias.zero_()

    def forward(self, x):

        x = self.backbone(x)
        if self.relu:
            x = torch.relu(x)
        x = torch.nn.functional.normalize(x)
        x = self.classifier(x)
        return x

def validate(model, data):

    model.eval()
    num = 0
    correct = 0
    with torch.no_grad():
        for x, y in data.valid_dl:
            yh = micro_forward(model, x, y)
            num += x.size(0)
            correct += (yh.argmax(dim=1) == y).sum().item()
    accuracy = 100. * correct / num
    return accuracy

cross_entropy = torch.nn.functional.cross_entropy

def entropy(yh, y):

    p = torch.nn.functional.softmax(yh, dim=1)
    log_p = torch.nn.functional.log_softmax(yh, dim=1)
    loss = - (p * log_p).sum(dim=1).mean()
    return loss

class Flatten(torch.nn.Module):

    def forward(self, x):

        x = x.view(x.size(0), -1)
        return x

class Conv64(torch.nn.Module):

    @staticmethod
    def conv_bn(in_channels, out_channels, kernel_size, padding, pool):

        model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(pool)
            )
        return model

    def __init__(self, num_classes, image_size):

        super().__init__()

        self.model = torch.nn.Sequential(
            self.conv_bn(3, 64, 3, 1, 2),
            self.conv_bn(64, 64, 3, 1, 2),
            self.conv_bn(64, 64, 3, 1, 2),
            self.conv_bn(64, 64, 3, 1, 2),
            Flatten(),
            torch.nn.Linear(64 * (int(image_size / 16) ** 2), num_classes)
            )

    def forward(self, x):

        x = self.model(x)
        return x

class ResNet12(torch.nn.Module):

    class Block(nn.Module):

        def __init__(self, in_channels, out_channels):

            super().__init__()

            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3,
                padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3,
                padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv3 = torch.nn.Conv2d(out_channels, out_channels, 3,
                padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.conv_res = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn_res = nn.BatchNorm2d(out_channels)
            self.maxpool = nn.MaxPool2d(2)
            self.relu = nn.ReLU()

        def forward(self, x):

            residual = self.conv_res(x)
            residual = self.bn_res(residual)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)

            x += residual
            x = self.relu(x)
            x = self.maxpool(x)

            return x

    def __init__(self, num_classes, image_size):

        super().__init__()

        self.model = torch.nn.Sequential(
            self.Block(3, 64),
            self.Block(64, 128),
            self.Block(128, 256),
            self.Block(256, 512),
            torch.nn.AvgPool2d(int(image_size / 16), stride=1),
            Flatten(),
            torch.nn.Linear(512, num_classes)
            )
        self.reset_parameters()

    def reset_parameters(self):

        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out',
                    nonlinearity='relu')

    def forward(self, x):

        x = self.model(x)
        return x

class WRN2810(torch.nn.Module):

    def __init__(self, num_classes, image_size):

        super().__init__()

        self.model = \
            partial(wrn, num_groups=3, N=4, k=10)(num_classes=num_classes)

    def forward(self, x):

        x = self.model(x)
        return x

