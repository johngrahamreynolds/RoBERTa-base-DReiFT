import datasets
import evaluate
# from evaluate.metrics.recall import Recall
from sklearn.metrics import recall_score

_DESCRIPTION = """
Custom built Recall metric that accepts underlying kwargs at instantiation time. 
This class allows one to circumvent the current issue of `combine`-ing the Recall metric, instantiated with its own parameters, into a `CombinedEvaluations` class with other metrics.
\n
In general, the recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
The recall is intuitively the ability of the classifier to find all the positive samples.
"""

_CITATION = """
@online{MarioBbqRec,
  author = {John Graham Reynolds aka @MarioBarbeque},
  title = {{Fixed Recall Hugging Face Metric},
  year = 2024,
  url = {https://huggingface.co/spaces/MarioBarbeque/FixedRecall},
  urldate = {2024-11-6}
}
"""

_INPUTS = """
'average': This parameter is required for multiclass/multilabel targets. 
If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data. 
Options include: {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or `None`. The default is `binary`.
"""

# could in principle subclass Recall, but ideally we can work the fix into the Recall class to maintain SOLID code
# for this immediate fix we create a new class

class FixedRecall(evaluate.Metric):

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
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html"],
        )
    
    # could remove specific kwargs like average, sample_weight from _compute() method and simply pass them to the underlying scikit-learn function in the form of a class var self.*
    # but leaving for sake of potentially subclassing Recall

    def _compute(
        self, predictions, references, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn",
    ):
        score = recall_score(
            references, predictions, labels=labels, pos_label=pos_label, average=self.average, sample_weight=sample_weight, zero_division=zero_division,
        )
        return {"recall": float(score) if score.size == 1 else score}