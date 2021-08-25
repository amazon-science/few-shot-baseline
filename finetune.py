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

from utils import *

def get_args():

    parser = argparse.ArgumentParser(description='Finetune',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='Mini-ImageNet')
    parser.add_argument('--architecture', type=str, default='WRN2810')
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--support_shot', type=int, default=1)
    parser.add_argument('--query_shot', type=int, default=15)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--train_and_val', action='store_true')
    parser.add_argument('--non_transductive', action='store_true')
    args = parser.parse_args()
    return args.dataset, args.architecture, args.way, args.support_shot, \
        args.query_shot, args.num_episodes, args.train_and_val, \
        args.non_transductive

def fine_tune(backbone, relu, lamb, support, query, non_transductive,
    hardness_model):

    '''
    Transductive Fine-Tuning
    The idea is to use information from the test datum to restrict the
    hypothesis space while searching for the classifier at test time. We
    introduce a regularizer on the test data as we seek outputs with a peaked
    posterior, or low Shannon Entropy.
    '''

    lr = 5e-5
    epochs = 25

    model = FewShotModel(backbone, support, relu)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    hardness = hardness_model.get_hardness(support, query)

    init_accuracy = validate(model, query)

    model.train()
    for epoch in range(1, epochs + 1):
        for (xs, ys), (xq, yq) in zip(support.train_dl, query.train_dl):
            break
        micro_forward(model, xs, ys, cross_entropy, 1.)
        optimizer.step()
        if not non_transductive:
            micro_forward(model, xq, yq, entropy, lamb)
            optimizer.step()

    final_accuracy = validate(model, query)

    return init_accuracy, final_accuracy, hardness

def main():

    set_seed()

    dataset, architecture, way, support_shot, query_shot, num_episodes, \
        train_and_val, non_transductive = get_args()
    params = get_params(dataset, architecture, train_and_val)
    relu = params['relu']
    lamb = params['lambda']
    image_size = params['image_size']
    csv_file_name = params['csv_file_name']
    parameter_path = os.path.join(dataset, 'models',
        params['parameter_file_name'] + '.pth')

    data = FewShotDataset(dataset, 'test.csv', image_size, way, support_shot,
        query_shot)

    meta_train_data = get_data(dataset, csv_file_name, csv_file_name,
        image_size, 1)
    meta_train_classes = len(meta_train_data.valid_ds.y.classes)
    backbone = globals()[architecture](num_classes=meta_train_classes,
        image_size=image_size)
    backbone.load_state_dict(torch.load(parameter_path)['model'])
    backbone = backbone.cuda()

    hardness_model = Hardness()

    results = {
        'dataset'           : dataset,
        'architecture'      : architecture,
        'way'               : way,
        'support_shot'      : support_shot,
        'query_shot'        : query_shot,
        'train_and_val'     : train_and_val,
        'non_transductive'  : non_transductive,
        'init_accuracy'     : [],
        'final_accuracy'    : [],
        'hardness'          : []
        }
    file_name = dataset + '_' + architecture + '_' + str(way) + '_' \
        + str(support_shot ) + '_' + str(query_shot) + '_' \
        + str(train_and_val) + '_' + str(non_transductive)
    os.makedirs('results', exist_ok=True)
    print('\t mean \t standard-deviation \t confidence-interval')
    for episode in tqdm(range(1, num_episodes + 1)):
        support, query = data[episode]
        init_accuracy, final_accuracy, hardness = fine_tune(backbone, relu,
            lamb, support, query, non_transductive, hardness_model)
        results['init_accuracy'].append(init_accuracy)
        results['final_accuracy'].append(final_accuracy)
        results['hardness'].append(hardness)
        tqdm.write('Init \t %2.2f \t\t %2.2f \t\t\t %2.2f' % (
            np.mean(results['init_accuracy']),
            np.std(results['init_accuracy']),
            1.96 * np.std(results['init_accuracy']) / (episode ** 0.5))
            )
        tqdm.write('Final \t %2.2f \t\t %2.2f \t\t\t %2.2f' % (
            np.mean(results['final_accuracy']),
            np.std(results['final_accuracy']),
            1.96 * np.std(results['final_accuracy']) / (episode ** 0.5))
            )
        torch.save(results, os.path.join('results', file_name + '.pth'))

if __name__ == '__main__':

    main()

