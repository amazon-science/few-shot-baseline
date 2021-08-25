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

matplotlib.rcParams.update({'font.size': 18})
sns.set_style('darkgrid')

def main():

    dataset = []
    init_accuracy = []
    final_accuracy = []
    hardness = []
    for file_name in os.listdir('results'):
        file_path = os.path.join('results', file_name)
        results = torch.load(file_path)
        num_episodes = len(results['hardness'])
        dataset += [results['dataset'] for _ in range(num_episodes)]
        init_accuracy += results['init_accuracy']
        final_accuracy += results['final_accuracy']
        hardness += results['hardness']
    results = {
        'dataset'           : dataset,
        'init_accuracy'     : init_accuracy,
        'final_accuracy'    : final_accuracy,
        'hardness'          : hardness
        }
    results = pd.DataFrame.from_dict(results)

    fig = plt.figure(1, figsize=(8, 8))
    plt.clf()
    ax = fig.add_subplot(111)

    for index, ds in enumerate(set(dataset)):
        subset = results.query('dataset == "%s"' % ds)
        color = sns.color_palette('deep')[index]
        sns.regplot(
            data=subset,
            x='hardness',
            y='final_accuracy',
            truncate=True,
            color=color,
            label=ds,
            scatter=True,
            line_kws=dict(lw=1.5),
            ax=ax
            )
        slope, intercept, _, _, _ = \
            stats.linregress(subset['hardness'], subset['final_accuracy'])
        auc = - intercept ** 2. / (2. * slope)
        print('Area under Curve \t %s \t %2.3f' % (ds, auc))

    plt.legend()
    plt.xticks([1, 2, 3, 4, 5])
    plt.xlabel('Hardness'); plt.ylabel('Accuracy (%)')
    plt.savefig('Hardness.pdf', bbox_inches='tight')

if __name__ == '__main__':

    main()

