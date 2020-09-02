import torch
from torch.nn.parallel.scatter_gather import scatter

import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

torch.is_tensor()

def main() -> None:
    output = scatter({'a': torch.tensor([1]), 'b': torch.tensor([3,4])}, [0, 1])
    
    instances = ((4, 62), (3, 63), (1, 55))
    for index, (batch_size, other) in enumerate(instances):
        num_choice = 4
        features = {
            'input_ids': torch.randint(0, 1000, (batch_size * num_choice, other)),
            'attention_mask': torch.randint(0, 2, (batch_size * num_choice, other)),
            'token_type_ids': torch.randint(0, 2, (batch_size * num_choice, other)),
            'labels': torch.randint(0, 5, (batch_size,)),
            'num_choice': torch.LongTensor([num_choice] * batch_size),
        }
        inputs = (features, index)
        # print(inputs)
        output = scatter(inputs, [0, 1])
        # print(output)
        pass


if __name__ == '__main__':
    main()
