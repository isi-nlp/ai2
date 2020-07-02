import itertools


def main() -> None:
    tasks = ('cycic_real_10',)
    agbs = (16, 32, 64)
    batches = (1, 2)
    lrs = ('5e-7', '1e-6', '2e-6', '5e-6', '1e-5', '5e-5')
    seeds = (42, 643723858, 1813174159)

    for task, agb, batch, lr, seed in itertools.product(tasks, agbs, batches, lrs, seeds):
        save_path = f'roberta-large-{task}-agb{agb}-bs{batch}-lr{lr}-s{seed}'
        params = [
            f'task={task}',
            f'accumulate_grad_batches={agb}',
            f'batch_size={batch}',
            f'learning_rate={lr}',
            f'random_seed={seed}',
            f'save_path={save_path}',
        ]
        print(' '.join(params))


if __name__ == '__main__':
    main()
