from collections import Counter
from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def plot_labels(datasets, id2label, version, output_dir: Path):
    output_dir.joinpath("fig").mkdir(exist_ok=True)

    for ds_name, dataset in datasets.items():
        if dataset is None:
            continue
        label_counter = Counter(dataset["label_ids"].tolist())
        bplot_data = {'label_ids': [id2label[int(l)] for l in sorted(label_counter)],
                      'occ': [label_counter[k] for k in sorted(label_counter)]}
        ax = sns.barplot(x="label_ids", y="occ", data=bplot_data)
        ax.set_title(f"Class distribution {ds_name} dataset")
        for p in ax.patches:
            ax.annotate(f"\n{int(p.get_height())}", (p.get_x() + 0.33, p.get_height()), ha="center", va="top", size=14)

        plt.savefig(output_dir.joinpath(f"fig/distribution_{version}_{ds_name}.png"))
        plt.savefig(output_dir.joinpath(f"fig/distribution_{version}_{ds_name}.eps"))
        plt.clf()


def plot_confusion_matrix(preds, gold_labels, labels, output_dir: Path, suff: str = None):
    output_dir.joinpath("fig").mkdir(exist_ok=True)

    for normalize in (None, "true"):
        suffix = ('_norm' if normalize else '') + (f'_{suff}' if suff else '')
        ConfusionMatrixDisplay.from_predictions(gold_labels, preds, display_labels=labels, normalize=normalize,
                                                cmap=sns.light_palette("seagreen", as_cmap=True))
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout(pad=0.5)
        plt.savefig(output_dir.joinpath(f"fig/test_confusion_matrix{suffix}.png"),
                    dpi=300,
                    bbox_inches="tight")
        plt.savefig(output_dir.joinpath(f"fig/test_confusion_matrix{suffix}.eps"),
                    bbox_inches="tight")
        plt.clf()
