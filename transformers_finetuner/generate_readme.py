import json
from os import PathLike
from pathlib import Path

from typing import Any, Dict, List, MutableMapping, Union

import yaml

STRUCTURE = {
    "Dataset": [
        "dataset_name",
        "dataset_config",
        "dataset_revision",
        "labelcolumn",
        "textcolumn",
    ],
    "Training": [
        "optim",
        "learning_rate",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "save_steps",
        "metric_for_best_model",
    ],
    "Best checkedpoint based on validation": [
        "best_metric",
        "best_model_checkpoint",
    ]
}


def rec_merge(d1, d2):
    '''
    https://stackoverflow.com/a/24088493/1150683
    Update two dicts of dicts recursively,
    if either mapping has leaves that are non-dicts,
    the second's leaf overwrites the first's.
    '''
    for k, v in d1.items():
        if k in d2:
            # this next check is the only difference!
            if all(isinstance(e, MutableMapping) for e in (v, d2[k])):
                d2[k] = rec_merge(v, d2[k])
            # we could further check types and merge as appropriate here.
    d3 = d1.copy()
    d3.update(d2)
    return d3

def format_dict(d: Dict[str, Any]):
    return "\n".join(f"- {k}: {v}" for k, v in d.items())


def get_modelcard(modelcard_base: Union[str, PathLike]):
    pmodelcard = Path(modelcard_base)
    modelcard = yaml.safe_load(pmodelcard.read_bytes())
    return modelcard


def generate_readme(din: Union[str, PathLike], modelcard_base: Union[str, PathLike], overwrite_readme: bool = False):
    pdin = Path(din).resolve()
    preadme = pdin / "README.md"
    mode = "w" if overwrite_readme else "a"

    test_results: Dict[str, float] = json.loads(pdin.joinpath("test_results.json").read_bytes())
    env: Dict[str, Any] = json.loads(pdin.joinpath("env.json").read_bytes())
    trainer_state: Dict[str, Any] = json.loads(pdin.joinpath("trainer_state.json").read_bytes())
    args: Dict[str, Any] = json.loads(pdin.joinpath("args.json").read_bytes())
    args = {**args, **trainer_state}

    with preadme.open(mode=mode, encoding="utf-8") as fhout:
        modelcard = get_modelcard(modelcard_base)
        modelcard["datasets"] = [args["dataset_name"]]
        modelcard["metrics"] = list(test_results.keys())
        model_index = {
            "name": f"{args['model_name_or_path'].split('/')[-1]}-{args['dataset_name'].split('/')[-1]}",
            "results": [{
                "dataset": {
                    "type": args["dataset_name"],
                    "name": f"{args['dataset_name']} - {args['dataset_config']} - {args['dataset_revision']}",
                    "config": args['dataset_config'],
                    "split": args["testsplit_name"],
                    "revision": args['dataset_revision']
                },
                "metrics": [{
                    "type": m,
                    "value": v,
                    "name": f"{args['testsplit_name'].capitalize()} {m}"
                } for m, v in test_results.items()]
            }]
        }
        modelcard["model-index"] = [rec_merge(modelcard["model-index"][0], model_index)]

        fhout.write(f"---\n{yaml.dump(modelcard)}---\n\n")

        fhout.write(f"# {args['model_name_or_path'].split('/')[-1]}-{args['dataset_name'].split('/')[-1]}\n\n")
        for title, keys in STRUCTURE.items():
            temp_d = {k: v for k, v in args.items() if k in keys}
            temp_d = {k: v for k, v in sorted(temp_d.items(), key=lambda k: keys.index(k[0]))}
            fhout.write(f"# {title}\n{format_dict(temp_d)}\n\n")

        fhout.write(f"# Test results of best checkpoint\n{format_dict(test_results)}\n\n")
        fhout.write(f"## Confusion matric\n\n")
        fhout.write("![cfm](fig/test_confusion_matrix.png)\n\n")
        fhout.write(f"## Normalized confusion matrix\n\n")
        fhout.write("![norm cfm](fig/test_confusion_matrix_norm.png)\n\n")
        fhout.write(f"# Environment\n{format_dict(env)}\n\n")

    print("Done! Verify the README before uploading!!!")


def main():
    import argparse

    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument("din", help="Input directory that contains the model and all output jsons."
                                     "Will also write README.md here.")
    cparser.add_argument("modelcard_base", help="Path to model card base YAML.")
    cparser.add_argument("-f", "--overwrite_readme", help="Overwrite a README if it already exists.",
                         action="store_true")

    cargs = cparser.parse_args()
    generate_readme(**vars(cargs))


if __name__ == "__main__":
    main()
