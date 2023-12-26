from datasets import load_metric
import numpy as np
import evaluate
from sklearn.metrics import f1_score
from transformers import EvalPrediction

# def compute_metrics(eval_pred):
#     # metric = load_metric("accuracy")
#     metric = evaluate.load("accuracy")
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     accuracy = metric.compute(predictions=predictions, references=labels)
#     f1 = f1_score(labels, predictions, average="weighted")
#     return {"accuracy": accuracy, "f1": f1}


# def compute_metrics(p: EvalPrediction):
#     metric = load_metric("glue", "cola")
#     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
#     preds = np.argmax(preds, axis=1)
#     result = metric.compute(predictions=preds, references=p.label_ids)
#     if len(result) > 1:
#         result["combined_score"] = np.mean(list(result.values())).item()
#     return result
def compute_metrics(eval_pred):
    metric = load_metric("glue", "mrpc")
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
