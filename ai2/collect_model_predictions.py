import json
import logging
from pathlib import Path
import string
from typing import Any, Dict, List
from zipfile import ZipFile

from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

logger = logging.getLogger(__name__)


def collect_model_predictions_entry_point(params: Parameters) -> None:
    project_root = params.existing_directory("project_root")
    experiment_root = params.existing_directory("experiment_root")
    save_zip_to = params.creatable_file("save_zip_to")

    physicaliqa_data_root = project_root / "task_data" / "physicaliqa-train-dev"
    socialiqa_data_root = project_root / "task_data" / "socialiqa-train-dev"

    logger.info("Loading task data...")
    task_to_data = {
        "physicaliqa": load_task_data(physicaliqa_data_root),
        "socialiqa": load_task_data(socialiqa_data_root),
    }
    recognized_tasks = tuple(task_to_data.keys())
    logger.info("Successfully loaded recognized task data.")

    logger.info("Zipping predictions...")
    prediction_files = list(experiment_root.rglob("predictions.lst"))
    with ZipFile(save_zip_to, "w") as zip_file:
        for idx, prediction_file in enumerate(prediction_files):
            task = None
            for recognized_task in recognized_tasks:
                if recognized_task in str(prediction_file):
                    task = recognized_task
                    break

            if task is None:
                raise RuntimeError(f"Don't recognize task for prediction file {prediction_file}! Recognized: {recognized_tasks}")

            # Save aligned predictions
            predictions = load_labels(prediction_file)
            aligned_predictions = [
                {**task_data, "predicted_label": predicted_label}
                for task_data, predicted_label in zip(task_to_data[task], predictions)
            ]
            for aligned_prediction in aligned_predictions:
                predicted_label = aligned_prediction["predicted_label"]
                gold_label = aligned_prediction["gold_label"]
                if task == "socialiqa":
                    aligned_prediction["predicted_label"] = human_readable_label(
                        predicted_label - 1
                    ).upper()
                    aligned_prediction["gold_label"] = human_readable_label(
                        gold_label - 1
                    ).upper()
                else:
                    aligned_prediction["predicted_label"] = human_readable_label(
                        predicted_label
                    )
                    aligned_prediction["gold_label"] = human_readable_label(
                        gold_label - 1
                    )

            aligned_zip_path = prediction_file.relative_to(experiment_root).with_suffix(".jsonl")
            zip_file.writestr(
                str(aligned_zip_path),
                "\n".join(json.dumps(aligned_prediction) for aligned_prediction in aligned_predictions) + "\n",
            )

            # Save unaligned predictions
            raw_zip_path = aligned_zip_path.with_name(aligned_zip_path.stem + "_raw.lst")
            zip_file.write(
                prediction_file.absolute(),
                arcname=str(raw_zip_path)
            )

            if (idx + 1) % 10 == 0:
                logger.info("Processed %d / %d prediction files.")
    logger.info("Processed %d prediction files to zip file %s.", len(prediction_files), save_zip_to)


def load_task_data(task_data_root: Path) -> List[Dict[str, Any]]:
    features_path = task_data_root / "dev.jsonl"
    labels_path = task_data_root / "dev-labels.lst"

    return [
        {**features_dict, "gold_label": label}
        for features_dict, label in zip(
            [json.loads(line) for line in features_path.read_text().splitlines()],
            load_labels(labels_path),
        )
    ]


def load_labels(labels_path: Path) -> List[int]:
    return [int(line) for line in labels_path.read_text().splitlines()]


def human_readable_label(label: int) -> str:
    return string.ascii_lowercase[label]


if __name__ == '__main__':
    parameters_only_entry_point(collect_model_predictions_entry_point)
