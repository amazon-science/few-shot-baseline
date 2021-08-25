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

import cv2
import math
import os
import pandas as pd
import pickle
import tqdm

def main():

    for data_type in ['train', 'train_val', 'val', 'test']:

        csv = {'fn' : [], 'c' : []}

        if data_type == 'train':
            pkl_file = 'miniImageNet_category_split_train_phase_train.pickle'
        elif data_type == 'train_val':
            pkl_file = 'miniImageNet_category_split_train_phase_val.pickle'
        else:
            pkl_file = 'miniImageNet_category_split_' + data_type + '.pickle'
        pkl_file = os.path.join('MiniImagenet', pkl_file)
        d = pickle.load(open(pkl_file, 'rb'), encoding='latin1')
        data = d['data']
        label_to_class = {label: cls for (cls, label) in d['catname2label'].items()}
        labels = d['labels']

        num_images = len(labels)
        num_digits = math.ceil(math.log(num_images, 10))

        for idx in tqdm.tqdm(range(num_images)):
            cls = label_to_class[labels[idx]]
            os.makedirs(os.path.join(data_type, cls), exist_ok=True)
            image = data[idx][:,:,::-1]
            image_path = os.path.join(data_type, cls, str(idx).zfill(num_digits) + '.png')
            cv2.imwrite(image_path, image)
            csv['fn'].append(image_path)
            csv['c'].append(cls)

        pd.DataFrame(csv).to_csv(os.path.join(data_type + '.csv'), index=False, columns=['fn', 'c'])

    train_and_val = pd.concat([pd.read_csv(f) for f in ['train.csv', 'val.csv']])
    train_and_val.to_csv('train_and_val.csv', index=False, columns=['fn', 'c'])

if __name__ == '__main__':

    main()

