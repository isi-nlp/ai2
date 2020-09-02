from pytorch_lightning import LightningModule
import torch


class Classifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.total = torch.nn.Parameter(torch.Tensor(1))
        self.other = torch.nn.Parameter(torch.Tensor(1))

    def forward(self):
        return


def main() -> None:
    classifier = Classifier()
    parameter_names = sorted(name for name, _ in classifier.named_parameters())
    print(parameter_names)


if __name__ == "__main__":
    main()
