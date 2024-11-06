import datasets
import evaluate
# from evaluate.metrics.f1 import F1
from sklearn.metrics import f1_score

_DESCRIPTION = """
Custom built F1 metric that accepts underlying kwargs at instantiation time. 
This class allows one to circumvent the current issue of `combine`-ing the f1 metric, instantiated with its own parameters, into a `CombinedEvaluations` class with other metrics.
\n
In general, the F1 score is the harmonic mean of the precision and recall. It can be computed with the equation:\n
F1 = 2 * (precision * recall) / (precision + recall)
"""

_CITATION = """
@online{MarioBbqF1,
  author = {John Graham Reynolds aka @MarioBarbeque},
  title = {{Fixed F1 Hugging Face Metric},
  year = 2024,
  url = {https://huggingface.co/spaces/MarioBarbeque/FixedF1},
  urldate = {2024-11-5}
}
"""

_INPUTS = """
'average': This parameter is required for multiclass/multilabel targets. 
If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data. 
Options include: {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or `None`. The default is `binary`.
"""

# could in principle subclass the F1 Metric, but ideally we can work the fix into HF evaluate's main F1 class to maintain SOLID code
# for this fix we create a new class

class FixedF1(evaluate.Metric):

    def __init__(self, average="binary"):
        super().__init__()
        self.average = average
        # additional values passed to compute() could and probably should (?) all be passed here so that the final computation is configured immediately at Metric instantiation

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_INPUTS,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"],
        )
    
    # could remove specific kwargs like average, sample_weight from _compute() method of F1
    # but leaving for sake of potentially subclassing F1

    def _compute(self, predictions, references, labels=None, pos_label=1, average="binary", sample_weight=None):
        score = f1_score(
            references, predictions, labels=labels, pos_label=pos_label, average=self.average, sample_weight=sample_weight
        )
        return {"f1": float(score) if score.size == 1 else score}