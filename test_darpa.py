import argparse
import warnings
import torch
from ai2.model import Classifier
from run_darpa import MODELS, CONFIGS, TOKENIZERS

warnings.simplefilter(action='ignore', category=FutureWarning)


def load_from_metrics(base, weights_path, on_gpu, map_location=None, **kargs):
    """
    Primary way of loading model from csv weights path
    :param weights_path:
    :param tags_csv:
    :param on_gpu:
    :param map_location: dic for mapping storage {'cuda:1':'cuda:0'}
    :return:
    """
    hparams = kargs
    hparams.__setattr__('on_gpu', on_gpu)

    if on_gpu:
        if map_location is not None:
            checkpoint = torch.load(weights_path, map_location=map_location)
        else:
            checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

    # load the state_dict on the model automatically
    model = base(hparams)
    model.load_state_dict(checkpoint['state_dict'])

    # give model a chance to load something
    model.on_load_checkpoint(checkpoint)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Eval ai2 darpa tasks with pytorch-transformers')
    parser.add_argument('--task', '-t', choices=['anli', 'hellaswag', 'physicaliqa', 'socialiqa'],
                        help='DARPA task, see https://leaderboard.allenai.org/?darpa_offset=0', required=True)

    parser.add_argument('--train_config', help='Training config file', required=True)
    parser.add_argument('--model_type', choices=MODELS, help='Model type', required=True)
    parser.add_argument('--tokenizer_type', choices=TOKENIZERS, help='Tokenizer type', required=True)
    parser.add_argument('--model_config_type', choices=CONFIGS, help='Model configuration type', required=False)
    parser.add_argument('--model_weight', help='Model weight from huggingface', required=True)
    parser.add_argument('--tokenizer_weight', help='Pretrained tokenizer from huggingface', required=True)
    parser.add_argument('--model_config_weight', help='Predefined configuration', required=False)
    parser.add_argument('--weights_path', help='Saved model weights file')
    parser.add_argument('--output', '-o', help='Output file')

    args = parser.parse_args()

    TASK = load_config("ai2/tasks.yaml", args.task)

    if args.model_config_weight is None or args.model_config_type is None:
        args.model_config_weight = args.model_weight
        args.model_config_type = args.model_type

    pretrained_model = load_from_metrics(
        base=Classifier,
        weights_path=args.weights_path,
        on_gpu=torch.cuda.is_available(),
        map_location=None,
        task_config=TASK,
        train_config=load_config(args.train_config),
        model_class=MODELS[args.model_type],
        model_path=args.model_weight,
        tokenizer_class=TOKENIZERS[args.tokenizer_type],
        tokenizer_path=args.tokenizer_weight,
        model_config_class=CONFIGS[args.model_config_type],
        model_config_path=args.model_config_weight
    )

    # predict
    pretrained_model.eval()
    pretrained_model.freeze()

    dataloader = pretrained_model.val_dataloader()

    outputs = []

    for i, batch in enumerate(dataloader):
        res = pretrained_model.validation_step(batch, i)
        outputs.append(res)

    truth = torch.cat([x['truth'] for x in outputs], dim=0).reshape(-1).cpu().detach().numpy().tolist()
    pred = torch.cat([x['pred'] for x in outputs], dim=0).reshape(-1).cpu().detach().numpy().tolist()

    assert truth == list(map(int, pretrained_model.dev_y))

    with open(args.output, "w") as output:
        output.write(f"Premise\tHypothesis\tTruth\tPrediction\n")
        for example, p in zip(pretrained_model.helper.preprocess(pretrained_model.dev_x, pretrained_model.dev_y), pred):
            for i, pair in enumerate(example.pairs):
                output.write(f"{pair.premise}\t{pair.hypothesis}\t{exmple.label == i}\t{p == i}\n")
