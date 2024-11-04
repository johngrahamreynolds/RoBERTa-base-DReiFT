import datasets
import evaluate
from evaluate import evaluator, Metric
# from evaluate.metrics.precision import Precision
from sklearn.metrics import precision_score

# could in principle subclass Precision, but ideally we can work the fix into the Precision class to maintain SOLID code
class FixedPrecision(evaluate.Metric):

    def __init__(self, average="binary", zero_division="warn"):
        super().__init__()
        self.average = average
        self.zero_division = zero_division
        # additional values passed to compute() could and probably should (?) all be passed here so that the final computation is configured immediately at Metric instantiation

    def _info(self):
        return evaluate.MetricInfo(
            description="Custom built Precision metric for true *multilabel* classification - the 'multilabel' config_name var in the evaluate.EvaluationModules class appears to better address multi-class classification, where features can fall under a multitude of labels. Granted, the subtlety is minimal and easily confused. This class is implemented with the intention of enabling the evaluation of multiple multilabel classification metrics at the same time using the evaluate.CombinedEvaluations.combine method.",
            citation="",
            inputs_description="'average': This parameter is required for multiclass/multilabel targets. If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data. Options include: {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or None.",
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

    def _compute(
        self, predictions, references, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn",
    ):
        score = precision_score(
            references, predictions, labels=labels, pos_label=pos_label, average=self.average, sample_weight=sample_weight, zero_division=self.zero_division,
        )
        return {"precision": float(score) if score.size == 1 else score}