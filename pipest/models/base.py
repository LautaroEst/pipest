import pytorch_lightning as pl
from torch import nn

from ..metrics.psr import binary_log_loss, log_loss, brier_score


class PSRClassifier(pl.LightningModule):

    name2func = {
        "normalized_binary_log_loss": lambda logits, labels: binary_log_loss(logits, labels, normalize=True),
        "normalized_log_loss": lambda logits, labels: log_loss(logits, labels, normalize=True),
        "normalized_brier_score": lambda logits, labels: brier_score(logits, labels, normalize=True),
        "binary_log_loss": lambda logits, labels: binary_log_loss(logits, labels, normalize=False),
        "log_loss": lambda logits, labels: log_loss(logits, labels, normalize=False),
        "brier_score": lambda logits, labels: brier_score(logits, labels, normalize=False),
    }

    def __init__(self, model, psr="cross_entropy", log_metrics=None):
        super().__init__()
        self._base_model = model
        self._loss_fn = self._parse_loss_fn(psr)
        self._log_metrics = self._parse_log_metrics(log_metrics)

    def _parse_log_metrics(self, log_metrics):
        if log_metrics is None:
            log_metrics = {}
        elif isinstance(log_metrics, str):
            log_metrics = {log_metrics: self.name2func[log_metrics]}
        elif isinstance(log_metrics, dict):
            for k, v in log_metrics.items():
                if isinstance(v, str):
                    log_metrics[k] = self.name2func[v]
                elif not callable(v):
                    raise ValueError("log_metrics must be str or callable")
        else:
            raise ValueError("log_metrics must be str or list of str")

    def _parse_loss_fn(self, loss):
        if loss == "cross_entropy":
            return nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        
    def forward(self, batch):
        output = self._base_model(batch)
        logits = output["logits"]
        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self._loss_fn(logits, batch["labels"])
        self.log("train_loss", loss)
        outputs = {"loss": loss}
        for k, v in self._log_metrics.items():
            self.log(f"train_{k}", v(logits, batch["labels"]))

        return outputs


