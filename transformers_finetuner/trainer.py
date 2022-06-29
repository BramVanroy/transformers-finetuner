from typing import Dict

from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    precision_recall_fscore_support, accuracy_score, \
    r2_score
from torch import FloatTensor
from torch.nn import CrossEntropyLoss
from transformers import Trainer

from utils import logger


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: FloatTensor, **kwargs):
        super().__init__(*args, **kwargs)

        class_weights = class_weights.to(self.args.device)
        logger.info(f"Using classification with class weights: {class_weights.tolist()}")
        self.loss_fct = CrossEntropyLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        try:
            loss = self.loss_fct(outputs.logits.view(-1, model.num_labels), labels.view(-1))
        except AttributeError:  # DataParallel
            loss = self.loss_fct(outputs.logits.view(-1, model.module.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred=None, label_ids=None, predictions=None, regression=False):
    """We want to be able to use this within the trainer, but also outside of it so. So the signature is
    a bit hacky:
        1. When used within the trainer, a partial is created beforehand with label_ids=None, predictions=None
        2. When used by itself, we do not use preds, but use label_ids and predictions instead
    :param pred:
    :param label_ids:
    :param predictions:
    :param regression:
    :return:
    """
    labels = pred.label_ids if pred is not None else label_ids
    if regression:
        preds = pred.predictions.squeeze() if pred is not None else predictions

        return {"mse": mean_squared_error(labels, preds).item(),
                "mae": mean_absolute_error(labels, preds).item(),
                "r2": r2_score(labels, preds).item()}
    else:
        preds = pred.predictions.argmax(-1) if pred is not None else predictions
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        return {
            "accuracy": accuracy_score(labels, preds).item(),
            "f1": f1.item(),
            "precision": precision.item(),
            "recall": recall.item()}


def compute_objective(metrics: Dict[str, float]) -> float:
    return metrics["eval_f1"]
