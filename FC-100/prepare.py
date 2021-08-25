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

    for data_type in ['train', 'val', 'test']:

        csv = {'fn' : [], 'c' : []}

        pkl_file = 'FC100_' + data_type + '.pickle'
        d = pickle.load(open(pkl_file, 'rb'), encoding='latin1')
        data = d['data']
        labels = d['labels']

        num_images = len(labels)
        num_digits = math.ceil(math.log(num_images, 10))

        for idx in tqdm.tqdm(range(num_images)):
            cls = str(labels[idx])
            os.makedirs(os.path.join(data_type, cls), exist_ok=True)
            image = data[idx][:,:,::-1]
            image_path = os.path.join(data_type, cls, str(idx).zfill(num_digits) + '.png')
            cv2.imwrite(image_path, image)
            csv['fn'].append(image_path)
            csv['c'].append(cls)

        pd.DataFrame(csv).to_csv(os.path.join(data_type + '.csv'), index=False, columns=['fn', 'c'])

        if data_type == 'train':
            pd.DataFrame(csv).to_csv(os.path.join('train_val.csv'), index=False, columns=['fn', 'c'])

    train_and_val = pd.concat([pd.read_csv(f) for f in ['train.csv', 'val.csv']])
    train_and_val.to_csv('train_and_val.csv', index=False, columns=['fn', 'c'])

if __name__ == '__main__':

    main()

