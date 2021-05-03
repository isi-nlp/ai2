import pandas as pd

from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point


def main(params: Parameters):
    model1_predicted_labels_path = params.existing_file('model1_predicted_labels')
    model2_predicted_labels_path = params.existing_file('model2_predicted_labels')
    gold_labels_path = params.existing_file('gold_labels')
    save_agreement_series_to = params.creatable_file('save_agreement_series_to')
    save_percent_agreement_to = params.creatable_file('save_percent_agreement_to')

    model1_predicted_labels: pd.Series = pd.read_csv(model1_predicted_labels_path, names=["label"])["label"]
    model2_predicted_labels: pd.Series = pd.read_csv(model2_predicted_labels_path, names=["label"])["label"]
    gold_labels: pd.Series = pd.read_csv(gold_labels_path, names=["label"])["label"]

    agreement_series: pd.Series = (model1_predicted_labels == gold_labels) == (model2_predicted_labels == gold_labels)
    agreement_series.to_csv(save_agreement_series_to, index=False, header=["both correct OR both incorrect?"])
    percent_agreement = agreement_series.mean()
    with save_percent_agreement_to.open("w") as file:
        print(f"{percent_agreement:.3}", file=file)


if __name__ == "__main__":
    parameters_only_entry_point(main)
