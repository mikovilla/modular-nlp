import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }

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
