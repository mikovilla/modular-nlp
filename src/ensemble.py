import numpy as np

class HFSkPipeEstimator:
    def __init__(self, pipe):
        self.pipe = pipe
        cfg = pipe.model.config
        
        if isinstance(cfg.id2label, dict) and len(cfg.id2label) > 0:
            ids = sorted([int(k) if isinstance(k, str) and k.isdigit() else int(k) for k in cfg.id2label.keys()])
            self.id2label = {i: str(cfg.id2label[i]) for i in ids}
            self.classes_ = np.array(ids)  # sklearn expects labels here; we’ll emit ints
        else:
            ids = list(range(cfg.num_labels))
            self.id2label = {i: str(i) for i in ids}
            self.classes_ = np.array(ids)

        self.label2id = {}
        
        for i, name in self.id2label.items():
            self.label2id[str(i)] = i
            self.label2id[f"LABEL_{i}"] = i
            self.label2id[name] = i

    def predict_proba(self, texts):
        if not isinstance(texts, list):
            texts = list(texts)
            
        texts = ["" if t is None else str(t) for t in texts]
        outs = self.pipe(texts, batch_size=32, truncation=True, return_all_scores=True)
        P = np.zeros((len(outs), len(self.classes_)), dtype=float)
        
        for i, row in enumerate(outs):
            for item in row:
                lbl = item["label"]
                if lbl in self.label2id:
                    j = self.label2id[lbl]
                    P[i, j] = float(item["score"])
                else:
                    if lbl.startswith("LABEL_") and lbl[6:].isdigit():
                        j = int(lbl[6:])
                        if j < P.shape[1]:
                            P[i, j] = float(item["score"])
        
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        
        return P / row_sums

    def predict(self, texts):
        P = self.predict_proba(texts)
        idx = P.argmax(axis=1)
        return idx