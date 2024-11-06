import datasets
import evaluate
# from evaluate.metrics.precision import Precision
from sklearn.metrics import precision_score

_DESCRIPTION = """
Custom built Precision metric that accepts underlying kwargs at instantiation time. 
This class allows one to circumvent the current issue of `combine`-ing the precision metric, instantiated with its own parameters, into a `CombinedEvaluations` class with other metrics.
\n
In general, the precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. 
The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
"""

_CITATION = """
@online{MarioBbqPrec,
  author = {John Graham Reynolds aka @MarioBarbeque},
  title = {{Fixed Precision Hugging Face Metric},
  year = 2024,
  url = {https://huggingface.co/spaces/MarioBarbeque/FixedPrecision},
  urldate = {2024-11-6}
}
"""

_INPUTS = """
'average': This parameter is required for multiclass/multilabel targets. 
If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data. 
Options include: {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or `None`. The default value for binary classification is `"binary"`.\n

'zero_division': "Sets the value to return when there is a zero division". Options include:
{`“warn”`, `0.0`, `1.0`, `np.nan`}. The default value is `"warn"`.
"""

# could in principle subclass Precision, but ideally we can work the fix into the Precision class to maintain SOLID code
# for this immediate fix we create a new class

class FixedPrecision(evaluate.Metric):

    def __init__(self, average="binary", zero_division="warn"):
        super().__init__()
        self.average = average
        self.zero_division = zero_division
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
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html"],
        )
    
    # could remove specific kwargs like average, sample_weight from _compute() method and simply pass them to the underlying scikit-learn function in the form of a class var self.*
    # but leaving for sake of potentially subclassing Precision

    def _compute(
        self, predictions, references, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn",
    ):
        score = precision_score(
            references, predictions, labels=labels, pos_label=pos_label, average=self.average, sample_weight=sample_weight, zero_division=self.zero_division,
        )
        return {"precision": float(score) if score.size == 1 else score}