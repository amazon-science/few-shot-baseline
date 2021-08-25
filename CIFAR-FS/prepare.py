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

import os
import pandas as pd
import tqdm

def main():

    for data_type in ['train', 'val', 'test']:

        txt_file = os.path.join('cifar100', 'splits', 'bertinetto', data_type + '.txt')
        classes = [line.strip() for line in open(txt_file)]

        image_path = os.path.join('cifar100', 'data')
        csv = {'fn' : [], 'c' : []}
        for idx in tqdm.tqdm(range(len(classes))):
            cls = classes[idx]
            images = os.listdir(os.path.join(image_path, cls))
            csv['fn'] += [os.path.join(image_path, cls, image) for image in images]
            csv['c'] += [cls for _ in images]

        pd.DataFrame(csv).to_csv(os.path.join(data_type + '.csv'), index=False, columns=['fn', 'c'])

        if data_type == 'train':
            pd.DataFrame(csv).to_csv(os.path.join('train_val.csv'), index=False, columns=['fn', 'c'])

    train_and_val = pd.concat([pd.read_csv(f) for f in ['train.csv', 'val.csv']])
    train_and_val.to_csv('train_and_val.csv', index=False, columns=['fn', 'c'])

if __name__ == '__main__':

    main()

