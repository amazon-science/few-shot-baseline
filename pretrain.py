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

@call_parse
def main(
    dataset: Param('Dataset', str)='Mini-ImageNet',
    architecture: Param('Architecture', str)='WRN2810',
    train_and_val: Param('Train and Val', bool)=False,
    gpu: Param('GPU', str)=None
    ):

    set_seed()

    params = get_params(dataset, architecture, train_and_val)
    batch_size = 256
    image_size = params['image_size']
    lr = 0.1
    lr_mult = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    cycle_length = params['cycle_length']
    num_cycles = 2
    cycle_multiplier = 2
    mixup = 0.25
    gpu = setup_distrib(gpu)
    num_gpus = num_distrib() or 1
    csv_file_name = params['csv_file_name']
    parameter_file_name = params['parameter_file_name']

    data = get_data(dataset, csv_file_name, 'train_val.csv', image_size,
        batch_size // num_gpus)

    num_classes = len(data.valid_ds.y.classes)
    model = globals()[architecture](num_classes, image_size)

    num_samples = len(data.train_dl) // num_gpus
    phases = [
        TrainingPhase(num_samples * cycle_length * (cycle_multiplier ** cycle))
            .schedule_hp('lr', lr * (lr_mult ** cycle), anneal=annealing_cos)
            .schedule_hp('mom', momentum)
            .schedule_hp('nesterov', True)
            .schedule_hp('wd', weight_decay)
        for cycle in range(num_cycles)
        ]
    epochs = int(cycle_length * (1 - (cycle_multiplier ** num_cycles)) \
        / (1 - cycle_multiplier))

    learn = Learner(
        data,
        model,
        metrics=[accuracy],
        bn_wd=False,
        true_wd=True,
        wd=weight_decay,
        loss_func=LabelSmoothingCrossEntropy(),
        opt_func=torch.optim.SGD
        )
    learn.mixup(alpha=mixup)
    learn.callback_fns += [
        partial(GeneralScheduler, phases=phases),
        partial(
            SaveModelCallback,
            every='improvement',
            monitor='accuracy',
            name=parameter_file_name
            )
        ]
    learn.to_distributed(gpu, cache_dir='/tmp')
    learn.to_fp16(dynamic=True)
    learn.fit(epochs)

