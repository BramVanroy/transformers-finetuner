import json
from os import PathLike
from pathlib import Path

from typing import List, Union


def fill_args(base: Union[str, PathLike], model_name_or_paths: Union[str, List[str]]):
    if isinstance(model_name_or_paths, str):
        model_name_or_paths = [model_name_or_paths]

    pbase = Path(base).resolve()
    for model_name_or_path in model_name_or_paths:
        args = json.loads(pbase.read_bytes())
        only_model_name = model_name_or_path.split("/")[-1]
        args["output_dir"] = f"{args['output_dir'].rstrip('/')}/{only_model_name}"
        args["model_name_or_path"] = model_name_or_path

        pout = pbase.with_name(f"{pbase.stem}_{only_model_name}.json")
        pout.write_text(json.dumps(args, indent=4), encoding="utf-8")


def main():
    import argparse

    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument("base", help="The base JSON args template to fill.")
    cparser.add_argument("-m", "--model_name_or_paths", help="model_name_or_path to fill in.", nargs="+")

    cargs = cparser.parse_args()
    fill_args(**vars(cargs))


if __name__ == "__main__":
    main()
