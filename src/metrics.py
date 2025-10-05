import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    f1_score, 
    mean_squared_error,
    precision_score, 
    recall_score
)

try:
    from transformers.trainer_utils import EvalPrediction
except Exception:
    EvalPrediction = None

def _unpack(eval_pred):
    # Supports (preds, labels) tuple or EvalPrediction
    if EvalPrediction is not None and isinstance(eval_pred, EvalPrediction):
        return eval_pred.predictions, eval_pred.label_ids
    return eval_pred  # assume (preds, labels)

def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def _softmax(logits):
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def compute_sse_mse_from_logits(logits, labels):
    """
    Brier-style: MSE between probabilities and one-hot labels.
    Returns (mse, sse).
    """
    logits = _to_numpy(logits)
    labels = _to_numpy(labels).astype(int)

    # If the model outputs class scores/logits: shape [N, C]
    if logits.ndim > 1:
        # Heuristic: if already probs (sum≈1), skip softmax
        row_sums = logits.sum(axis=-1, keepdims=True)
        if np.allclose(row_sums, 1.0, atol=1e-3) and np.all(logits >= 0.0):
            probs = logits
        else:
            probs = _softmax(logits)
        num_classes = probs.shape[-1]
        one_hot = np.eye(num_classes, dtype=float)[labels]
        mse = mean_squared_error(one_hot.reshape(-1), probs.reshape(-1))
        sse = mse * labels.shape[0]
        return mse, sse
    else:
        # Binary case with a single logit/prob per example
        # If values look like probs, use directly; else sigmoid is needed.
        x = logits
        if np.all((0.0 <= x) & (x <= 1.0)):
            probs_pos = x
        else:
            # sigmoid
            probs_pos = 1.0 / (1.0 + np.exp(-x))
        probs = np.stack([1.0 - probs_pos, probs_pos], axis=-1)
        one_hot = np.eye(2, dtype=float)[labels]
        mse = mean_squared_error(one_hot.reshape(-1), probs.reshape(-1))
        sse = mse * labels.shape[0]
        return mse, sse

def compute_metrics(eval_pred):
    logits, labels = _unpack(eval_pred)
    logits = _to_numpy(logits)
    labels = _to_numpy(labels).astype(int)

    # Predicted class ids
    if logits.ndim > 1:
        preds = np.argmax(logits, axis=-1)
    else:
        # Binary single logit/prob
        preds = (logits > 0.5).astype(int)

    # Classification metrics
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)

    # Probability-based SSE/MSE (Brier)
    mse, sse = compute_sse_mse_from_logits(logits, labels)

    return {
        "accuracy": acc,
        "f1_macro": f1m,
        "precision_macro": prec,
        "recall_macro": rec,
        "mse": mse,   # Brier
        "sse": sse
    }

def graph_log(history):
    df = pd.DataFrame(history)
    df_eval = df.dropna(subset=['eval_mse'])
    df_eval = df_eval.sort_values('epoch')
    plt.plot(df_eval['epoch'], df_eval['eval_mse'], label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Convergence Curve')
    plt.legend()
    plt.show()

def evaluate_pipe(pipe, texts, labels, id2label=None, average="macro"):
    if not isinstance(texts, list):
        texts = list(texts)
    clean_texts = []
    for x in texts:
        if x is None:
            clean_texts.append("")
        elif hasattr(x, "as_py"):
            clean_texts.append(str(x.as_py()))
        else:
            clean_texts.append(str(x))

    outs = pipe(clean_texts, batch_size=32, truncation=True, return_all_scores=True)
    pred_labels = [max(row, key=lambda r: r["score"])["label"] for row in outs]

    preds = []
    if isinstance(labels[0], int):
        if id2label is None and hasattr(pipe.model, "config"):
            id2label = getattr(pipe.model.config, "id2label", None)
        if id2label:
            label2id = {str(v): int(k) for k, v in id2label.items()}
        else:
            label2id = {}

        for lbl in pred_labels:
            if isinstance(lbl, int):
                preds.append(lbl)
            elif isinstance(lbl, str):
                if lbl in label2id:
                    preds.append(label2id[lbl])
                elif lbl.startswith("LABEL_") and lbl[6:].isdigit():
                    preds.append(int(lbl[6:]))
                elif lbl.isdigit():
                    preds.append(int(lbl))
                else:
                    raise ValueError(f"Unrecognized label format: {lbl}")
            else:
                raise ValueError(f"Unsupported label type: {type(lbl)}")
    else:
        preds = pred_labels

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average=average)
    rep = classification_report(labels, preds, digits=3)

    return {"accuracy": acc, f"f1_{average}": f1, "report": rep}
