# Copyright (c) Facebook, Inc. and its affiliates.
"""
The metrics module contains implementations of various metrics used commonly to
understand how well our models are performing. For e.g. accuracy, vqa_accuracy,
r@1 etc.

For implementing your own metric, you need to follow these steps:

1. Create your own metric class and inherit ``BaseMetric`` class.
2. In the ``__init__`` function of your class, make sure to call
   ``super().__init__('name')`` where 'name' is the name of your metric. If
   you require any parameters in your ``__init__`` function, you can use
   keyword arguments to represent them and metric constructor will take care of
   providing them to your class from config.
3. Implement a ``calculate`` function which takes in ``SampleList`` and
   `model_output` as input and return back a float tensor/number.
4. Register your metric with a key 'name' by using decorator,
   ``@registry.register_metric('name')``.

Example::

    import torch

    from mmf.common.registry import registry
    from mmf.modules.metrics import BaseMetric

    @registry.register_metric("some")
    class SomeMetric(BaseMetric):
        def __init__(self, some_param=None):
            super().__init__("some")
            ....

        def calculate(self, sample_list, model_output):
            metric = torch.tensor(2, dtype=torch.float)
            return metric

Example config for above metric::

    model_config:
        pythia:
            metrics:
            - type: some
              params:
                some_param: a
"""

import collections

import torch
from mmf.common.registry import registry
from mmf.datasets.processors.processors import EvalAIAnswerProcessor
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def _convert_to_one_hot(expected, output):
    # This won't get called in case of multilabel, only multiclass or binary
    # as multilabel will anyways be multi hot vector
    if output.squeeze().dim() != expected.squeeze().dim() and expected.dim() == 1:
        expected = torch.nn.functional.one_hot(
            expected.long(), num_classes=output.size(-1)
        ).float()
    return expected


class Metrics:
    """Internally used by MMF, Metrics acts as wrapper for handling
    calculation of metrics over various metrics specified by the model in
    the config. It initializes all of the metrics and when called it runs
    calculate on each of them one by one and returns back a dict with proper
    naming back. For e.g. an example dict returned by Metrics class:
    ``{'val/vqa_accuracy': 0.3, 'val/r@1': 0.8}``

    Args:
        metric_list (ListConfig): List of DictConfigs where each DictConfig
                                        specifies name and parameters of the
                                        metrics used.
    """

    def __init__(self, metric_list):
        if not isinstance(metric_list, collections.abc.Sequence):
            metric_list = [metric_list]

        self.metrics = self._init_metrics(metric_list)

    def _init_metrics(self, metric_list):
        metrics = {}
        self.required_params = {"dataset_name", "dataset_type"}
        for metric in metric_list:
            params = {}
            dataset_names = []
            if isinstance(metric, collections.abc.Mapping):
                if "type" not in metric:
                    raise ValueError(
                        f"Metric {metric} needs to have 'type' attribute "
                        + "or should be a string"
                    )
                metric_type = key = metric.type
                params = getattr(metric, "params", {})
                # Support cases where uses need to give custom metric name
                if "key" in metric:
                    key = metric.key

                # One key should only be used once
                if key in metrics:
                    raise RuntimeError(
                        f"Metric with type/key '{metric_type}' has been defined more "
                        + "than once in metric list."
                    )

                # a custom list of dataset where this metric will be applied
                if "datasets" in metric:
                    dataset_names = metric.datasets
            else:
                if not isinstance(metric, str):
                    raise TypeError(
                        "Metric {} has inappropriate type"
                        "'dict' or 'str' allowed".format(metric)
                    )
                metric_type = key = metric

            metric_cls = registry.get_metric_class(metric_type)
            if metric_cls is None:
                raise ValueError(
                    f"No metric named {metric_type} registered to registry"
                )

            metric_instance = metric_cls(**params)
            metric_instance.name = key
            metric_instance.set_applicable_datasets(dataset_names)

            metrics[key] = metric_instance
            self.required_params.update(metrics[key].required_params)

        return metrics

    def __call__(self, sample_list, model_output, *args, **kwargs):
        values = {}

        dataset_type = sample_list.dataset_type
        dataset_name = sample_list.dataset_name

        with torch.no_grad():
            for metric_name, metric_object in self.metrics.items():
                if not metric_object.is_dataset_applicable(dataset_name):
                    continue
                key = f"{dataset_type}/{dataset_name}/{metric_name}"
                values[key] = metric_object._calculate_with_checks(
                    sample_list, model_output, *args, **kwargs
                )

                if not isinstance(values[key], torch.Tensor):
                    values[key] = torch.tensor(values[key], dtype=torch.float)
                else:
                    values[key] = values[key].float()

                if values[key].dim() == 0:
                    values[key] = values[key].view(1)

        registry.register(
            "{}.{}.{}".format("metrics", sample_list.dataset_name, dataset_type), values
        )

        return values


class BaseMetric:
    """Base class to be inherited by all metrics registered to MMF. See
    the description on top of the file for more information. Child class must
    implement ``calculate`` function.

    Args:
        name (str): Name of the metric.

    """

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.required_params = ["scores", "targets"]
        # the set of datasets where this metric will be applied
        # an empty set means it will be applied on *all* datasets
        self._dataset_names = set()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Abstract method to be implemented by the child class. Takes
        in a ``SampleList`` and a dict returned by model as output and
        returns back a float tensor/number indicating value for this metric.

        Args:
            sample_list (SampleList): SampleList provided by the dataloader for the
                                current iteration.
            model_output (Dict): Output dict from the model for the current
                                 SampleList

        Returns:
            torch.Tensor|float: Value of the metric.

        """
        # Override in your child class
        raise NotImplementedError("'calculate' must be implemented in the child class")

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def _calculate_with_checks(self, *args, **kwargs):
        value = self.calculate(*args, **kwargs)
        return value

    def set_applicable_datasets(self, dataset_names):
        self._dataset_names = set(dataset_names)

    def is_dataset_applicable(self, dataset_name):
        return len(self._dataset_names) == 0 or dataset_name in self._dataset_names


@registry.register_metric("accuracy")
class Accuracy(BaseMetric):
    """Metric for calculating accuracy.

    **Key:** ``accuracy``
    """

    def __init__(self):
        super().__init__("accuracy")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: accuracy.

        """
        output = model_output["scores"]
        expected = sample_list["targets"]

        assert (
            output.dim() <= 2
        ), "Output from model shouldn't have more than dim 2 for accuracy"
        assert (
            expected.dim() <= 2
        ), "Expected target shouldn't have more than dim 2 for accuracy"

        if output.dim() == 2:
            output = torch.max(output, 1)[1]

        # If more than 1
        # If last dim is 1, we directly have class indices
        if expected.dim() == 2 and expected.size(-1) != 1:
            expected = torch.max(expected, 1)[1]

        correct = (expected == output.squeeze()).sum().float()
        total = len(expected)

        value = correct / total
        return value


@registry.register_metric("caption_bleu4")
class CaptionBleu4Metric(BaseMetric):
    """Metric for calculating caption accuracy using BLEU4 Score.

    **Key:** ``caption_bleu4``
    """

    def __init__(self):
        import nltk.translate.bleu_score as bleu_score

        self._bleu_score = bleu_score
        super().__init__("caption_bleu4")
        self.caption_processor = registry.get("coco_caption_processor")
        self.required_params = ["scores", "answers", "captions"]

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: bleu4 score.

        """
        # Create reference and hypotheses captions.
        references = []
        hypotheses = []

        # References
        targets = sample_list.answers
        for j, _ in enumerate(targets):
            img_captions = [
                self.caption_processor(c)["tokens"] for c in targets[j].tolist()
            ]
            references.append(img_captions)

        # Hypotheses
        if "captions" in model_output:
            scores = model_output["captions"]
        else:
            scores = torch.max(model_output["scores"], dim=-1)[1]
        scores = scores.tolist()
        predictions = []
        for j, _ in enumerate(scores):
            caption = self.caption_processor(scores[j])["tokens"]
            predictions.append(caption)
        hypotheses.extend(predictions)

        assert len(references) == len(hypotheses)

        bleu4 = self._bleu_score.corpus_bleu(references, hypotheses)

        return targets.new_tensor(bleu4, dtype=torch.float)


@registry.register_metric("vqa_accuracy")
class VQAAccuracy(BaseMetric):
    """
    Calculate VQAAccuracy. Find more information here_

    **Key**: ``vqa_accuracy``.

    .. _here: https://visualqa.org/evaluation.html
    """

    def __init__(self):
        super().__init__("vqa_accuracy")

    def _masked_unk_softmax(self, x, dim, mask_idx):
        x1 = torch.nn.functional.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate vqa accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: VQA Accuracy

        """
        output = model_output["scores"]
        # for three branch movie+mcan model
        if output.dim() == 3:
            output = output[:, 0]
        expected = sample_list["targets"]

        output = self._masked_unk_softmax(output, 1, 0)
        output = output.argmax(dim=1)  # argmax

        one_hots = expected.new_zeros(*expected.size())
        one_hots.scatter_(1, output.view(-1, 1), 1)
        scores = one_hots * expected
        accuracy = torch.sum(scores) / expected.size(0)

        return accuracy


@registry.register_metric("vqa_evalai_accuracy")
class VQAEvalAIAccuracy(BaseMetric):
    """
    Calculate Eval AI VQAAccuracy. Find more information here_
    This is more accurate and similar comparision to Eval AI
    but is slower compared to vqa_accuracy.

    **Key**: ``vqa_evalai_accuracy``.

    .. _here: https://visualqa.org/evaluation.html
    """

    def __init__(self):
        super().__init__("vqa_evalai_accuracy")
        self.evalai_answer_processor = EvalAIAnswerProcessor()
        self.required_params = ["scores", "answers", "context_tokens"]

    def _masked_unk_softmax(self, x, dim, mask_idx):
        x1 = torch.nn.functional.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate vqa accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: VQA Accuracy

        """
        output = model_output["scores"]
        expected = sample_list["answers"]

        answer_processor = registry.get(sample_list.dataset_name + "_answer_processor")
        answer_space_size = answer_processor.get_true_vocab_size()

        output = self._masked_unk_softmax(output, 1, 0)
        output = output.argmax(dim=1).clone().tolist()
        accuracy = []

        for idx, answer_id in enumerate(output):
            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = sample_list["context_tokens"][idx][answer_id]
            else:
                answer = answer_processor.idx2word(answer_id)

            answer = self.evalai_answer_processor(answer)

            gt_answers = [self.evalai_answer_processor(x) for x in expected[idx]]
            gt_answers = list(enumerate(gt_answers))

            gt_acc = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [item for item in other_answers if item[1] == answer]
                acc = min(1, float(len(matching_answers)) / 3)
                gt_acc.append(acc)
            avgGTAcc = float(sum(gt_acc)) / len(gt_acc)
            accuracy.append(avgGTAcc)

        accuracy = float(sum(accuracy)) / len(accuracy)

        return model_output["scores"].new_tensor(accuracy, dtype=torch.float)


class RecallAtK(BaseMetric):
    def __init__(self, name="recall@k"):
        super().__init__(name)

    def score_to_ranks(self, scores):
        # sort in descending order - largest score gets highest rank
        sorted_ranks, ranked_idx = scores.sort(1, descending=True)

        # convert from ranked_idx to ranks
        ranks = ranked_idx.clone().fill_(0)
        for i in range(ranked_idx.size(0)):
            for j in range(100):
                ranks[i][ranked_idx[i][j]] = j
        ranks += 1
        return ranks

    def get_gt_ranks(self, ranks, ans_ind):
        _, ans_ind = ans_ind.max(dim=1)
        ans_ind = ans_ind.view(-1)
        gt_ranks = torch.LongTensor(ans_ind.size(0))

        for i in range(ans_ind.size(0)):
            gt_ranks[i] = int(ranks[i, ans_ind[i].long()])
        return gt_ranks

    def get_ranks(self, sample_list, model_output, *args, **kwargs):
        output = model_output["scores"]
        expected = sample_list["targets"]

        ranks = self.score_to_ranks(output)
        gt_ranks = self.get_gt_ranks(ranks, expected)

        ranks = self.process_ranks(gt_ranks)
        return ranks.float()

    def calculate(self, sample_list, model_output, k, *args, **kwargs):
        ranks = self.get_ranks(sample_list, model_output)
        recall = float(torch.sum(torch.le(ranks, k))) / ranks.size(0)
        return recall


@registry.register_metric("r@1")
class RecallAt1(RecallAtK):
    """
    Calculate Recall@1 which specifies how many time the chosen candidate
    was rank 1.

    **Key**: ``r@1``.
    """

    def __init__(self):
        super().__init__("r@1")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Recall@1 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: Recall@1

        """
        return self.calculate(sample_list, model_output, k=1)


@registry.register_metric("r@5")
class RecallAt5(RecallAtK):
    """
    Calculate Recall@5 which specifies how many time the chosen candidate
    was among first 5 rank.

    **Key**: ``r@5``.
    """

    def __init__(self):
        super().__init__("r@5")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Recall@5 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: Recall@5

        """
        return self.calculate(sample_list, model_output, k=5)


@registry.register_metric("r@10")
class RecallAt10(RecallAtK):
    """
    Calculate Recall@10 which specifies how many time the chosen candidate
    was among first 10 ranks.

    **Key**: ``r@10``.
    """

    def __init__(self):
        super().__init__("r@10")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Recall@10 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: Recall@10

        """
        return self.calculate(sample_list, model_output, k=10)


@registry.register_metric("mean_r")
class MeanRank(RecallAtK):
    """
    Calculate MeanRank which specifies what was the average rank of the chosen
    candidate.

    **Key**: ``mean_r``.
    """

    def __init__(self):
        super().__init__("mean_r")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Mean Rank and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: mean rank

        """
        ranks = self.get_ranks(sample_list, model_output)
        return torch.mean(ranks)


@registry.register_metric("mean_rr")
class MeanReciprocalRank(RecallAtK):
    """
    Calculate reciprocal of mean rank..

    **Key**: ``mean_rr``.
    """

    def __init__(self):
        super().__init__("mean_rr")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Mean Reciprocal Rank and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: Mean Reciprocal Rank

        """
        ranks = self.get_ranks(sample_list, model_output)
        return torch.mean(ranks.reciprocal())


@registry.register_metric("textvqa_accuracy")
class TextVQAAccuracy(BaseMetric):
    def __init__(self):
        super().__init__("textvqa_accuracy")
        import mmf.utils.m4c_evaluators as evaluators

        self.evaluator = evaluators.TextVQAAccuracyEvaluator()
        self.required_params = ["scores", "answers", "context_tokens"]
        self.gt_key = "answers"

    def calculate(self, sample_list, model_output, *args, **kwargs):
        answer_processor = registry.get(sample_list.dataset_name + "_answer_processor")

        batch_size = sample_list.context_tokens.size(0)
        pred_answers = model_output["scores"].argmax(dim=-1)
        context_tokens = sample_list.context_tokens.cpu().numpy()
        answers = sample_list.get(self.gt_key).cpu().numpy()
        answer_space_size = answer_processor.get_true_vocab_size()

        predictions = []
        from mmf.utils.distributed import byte_tensor_to_object
        from mmf.utils.text import word_tokenize

        for idx in range(batch_size):
            tokens = byte_tensor_to_object(context_tokens[idx])
            answer_words = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(word_tokenize(tokens[answer_id]))
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )

            pred_answer = " ".join(answer_words).replace(" 's", "'s")
            gt_answers = byte_tensor_to_object(answers[idx])
            predictions.append({"pred_answer": pred_answer, "gt_answers": gt_answers})

        accuracy = self.evaluator.eval_pred_list(predictions)
        accuracy = torch.tensor(accuracy).to(sample_list.context_tokens.device)

        return accuracy


@registry.register_metric("stvqa_anls")
class STVQAANLS(TextVQAAccuracy):
    def __init__(self):
        super().__init__()
        self.name = "stvqa_anls"
        import mmf.utils.m4c_evaluators as evaluators

        self.evaluator = evaluators.STVQAANLSEvaluator()


@registry.register_metric("stvqa_accuracy")
class STVQAAccuracy(TextVQAAccuracy):
    def __init__(self):
        super().__init__()
        self.name = "stvqa_accuracy"
        import mmf.utils.m4c_evaluators as evaluators

        self.evaluator = evaluators.STVQAAccuracyEvaluator()


@registry.register_metric("ocrvqa_accuracy")
class OCRVQAAccuracy(STVQAAccuracy):
    def __init__(self):
        super().__init__()
        # same as STVQAAccuracy except for the name
        self.name = "ocrvqa_accuracy"


@registry.register_metric("textcaps_bleu4")
class TextCapsBleu4(TextVQAAccuracy):
    def __init__(self):
        super().__init__()
        self.name = "textcaps_bleu4"
        self.required_params = ["scores", "ref_strs", "context_tokens"]
        self.gt_key = "ref_strs"
        import mmf.utils.m4c_evaluators as evaluators

        self.evaluator = evaluators.TextCapsBleu4Evaluator()


@registry.register_metric("f1")
class F1(BaseMetric):
    """Metric for calculating F1. Can be used with type and params
    argument for customization. params will be directly passed to sklearn
    f1 function.
    **Key:** ``f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__("f1")
        self._multilabel = kwargs.pop("multilabel", False)
        self._sk_kwargs = kwargs

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate f1 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: f1.
        """
        scores = model_output["scores"]
        expected = sample_list["targets"]

        if self._multilabel:
            output = torch.sigmoid(scores)
            output = torch.round(output)
            expected = _convert_to_one_hot(expected, output)
        else:
            # Multiclass, or binary case
            output = scores.argmax(dim=-1)
            if expected.dim() != 1:
                # Probably one-hot, convert back to class indices array
                expected = expected.argmax(dim=-1)

        value = f1_score(expected.cpu(), output.cpu(), **self._sk_kwargs)

        return expected.new_tensor(value, dtype=torch.float)


@registry.register_metric("macro_f1")
class MacroF1(F1):
    """Metric for calculating Macro F1.

    **Key:** ``macro_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="macro", **kwargs)
        self.name = "macro_f1"


@registry.register_metric("micro_f1")
class MicroF1(F1):
    """Metric for calculating Micro F1.

    **Key:** ``micro_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "micro_f1"


@registry.register_metric("binary_f1")
class BinaryF1(F1):
    """Metric for calculating Binary F1.

    **Key:** ``binary_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", labels=[1], **kwargs)
        self.name = "binary_f1"


@registry.register_metric("multilabel_f1")
class MultiLabelF1(F1):
    """Metric for calculating Multilabel F1.

    **Key:** ``multilabel_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(multilabel=True, **kwargs)
        self.name = "multilabel_f1"


@registry.register_metric("multilabel_micro_f1")
class MultiLabelMicroF1(MultiLabelF1):
    """Metric for calculating Multilabel Micro F1.

    **Key:** ``multilabel_micro_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "multilabel_micro_f1"


@registry.register_metric("multilabel_macro_f1")
class MultiLabelMacroF1(MultiLabelF1):
    """Metric for calculating Multilabel Macro F1.

    **Key:** ``multilabel_macro_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="macro", **kwargs)
        self.name = "multilabel_macro_f1"


@registry.register_metric("roc_auc")
class ROC_AUC(BaseMetric):
    """Metric for calculating ROC_AUC.
    See more details at `sklearn.metrics.roc_auc_score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_ # noqa

    **Note**: ROC_AUC is not defined when expected tensor only contains one
    label. Make sure you have both labels always or use it on full val only

    **Key:** ``roc_auc``
    """

    def __init__(self, *args, **kwargs):
        super().__init__("roc_auc")
        self._sk_kwargs = kwargs

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate ROC_AUC and returns it back. The function performs softmax
        on the logits provided and then calculated the ROC_AUC.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "scores"
                                 field pointing to logits returned from the model.

        Returns:
            torch.FloatTensor: ROC_AUC.

        """

        output = torch.nn.functional.softmax(model_output["scores"], dim=-1)
        expected = sample_list["targets"]
        expected = _convert_to_one_hot(expected, output)
        value = roc_auc_score(expected.cpu(), output.cpu(), **self._sk_kwargs)
        return expected.new_tensor(value, dtype=torch.float)


@registry.register_metric("micro_roc_auc")
class MicroROC_AUC(ROC_AUC):
    """Metric for calculating Micro ROC_AUC.

    **Key:** ``micro_roc_auc``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "micro_roc_auc"


@registry.register_metric("macro_roc_auc")
class MacroROC_AUC(ROC_AUC):
    """Metric for calculating Macro ROC_AUC.

    **Key:** ``macro_roc_auc``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="macro", **kwargs)
        self.name = "macro_roc_auc"


@registry.register_metric("ap")
class AveragePrecision(BaseMetric):
    """Metric for calculating Average Precision.
    See more details at `sklearn.metrics.average_precision_score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score>`_ # noqa
    If you are looking for binary case, please take a look at binary_ap
    **Key:** ``ap``
    """

    def __init__(self, *args, **kwargs):
        super().__init__("ap")
        self._sk_kwargs = kwargs

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate AP and returns it back. The function performs softmax
        on the logits provided and then calculated the AP.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "scores"
                                 field pointing to logits returned from the model.

        Returns:
            torch.FloatTensor: AP.

        """

        output = torch.nn.functional.softmax(model_output["scores"], dim=-1)
        expected = sample_list["targets"]
        expected = _convert_to_one_hot(expected, output)
        value = average_precision_score(expected.cpu(), output.cpu(), **self._sk_kwargs)
        return expected.new_tensor(value, dtype=torch.float)


@registry.register_metric("binary_ap")
class BinaryAP(AveragePrecision):
    """Metric for calculating Binary Average Precision.
    See more details at `sklearn.metrics.average_precision_score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score>`_ # noqa
    **Key:** ``binary_ap``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.name = "binary_ap"

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Binary AP and returns it back. The function performs softmax
        on the logits provided and then calculated the binary AP.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "scores"
                                 field pointing to logits returned from the model.

        Returns:
            torch.FloatTensor: AP.

        """

        output = torch.nn.functional.softmax(model_output["scores"], dim=-1)
        # Take the score for positive (1) label
        output = output[:, 1]
        expected = sample_list["targets"]

        # One hot format -> Labels
        if expected.dim() == 2:
            expected = expected.argmax(dim=1)

        value = average_precision_score(expected.cpu(), output.cpu(), **self._sk_kwargs)
        return expected.new_tensor(value, dtype=torch.float)


@registry.register_metric("micro_ap")
class MicroAP(AveragePrecision):
    """Metric for calculating Micro Average Precision.

    **Key:** ``micro_ap``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "micro_ap"


@registry.register_metric("macro_ap")
class MacroAP(AveragePrecision):
    """Metric for calculating Macro Average Precision.

    **Key:** ``macro_ap``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="macro", **kwargs)
        self.name = "macro_ap"


@registry.register_metric("r@pk")
class RecallAtPrecisionK(BaseMetric):
    """Metric for calculating recall when precision is above a
    particular threshold. Use `p_threshold` param to specify the
    precision threshold i.e. k. Accepts precision in both 0-1
    and 1-100 format.

    **Key:** ``r@pk``
    """

    def __init__(self, p_threshold, *args, **kwargs):
        """Initialization function recall @ precision k

        Args:
            p_threshold (float): Precision threshold
        """
        super().__init__(name="r@pk")
        self.name = "r@pk"
        self.p_threshold = p_threshold if p_threshold < 1 else p_threshold / 100

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Recall at precision k and returns it back. The function
        performs softmax on the logits provided and then calculated the metric.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "scores"
                                 field pointing to logits returned from the model.

        Returns:
            torch.FloatTensor: Recall @ precision k.

        """
        output = torch.nn.functional.softmax(model_output["scores"], dim=-1)[:, 1]
        expected = sample_list["targets"]

        # One hot format -> Labels
        if expected.dim() == 2:
            expected = expected.argmax(dim=1)

        precision, recall, thresh = precision_recall_curve(expected.cpu(), output.cpu())

        try:
            value, _ = max(
                (r, p) for p, r in zip(precision, recall) if p >= self.p_threshold
            )
        except ValueError:
            value = 0

        return expected.new_tensor(value, dtype=torch.float)


@registry.register_metric("detection_mean_ap")
class DetectionMeanAP(BaseMetric):
    """Metric for calculating the detection mean average precision (mAP) using the COCO
    evaluation toolkit, returning the default COCO-style mAP@IoU=0.50:0.95

    **Key:** ``detection_mean_ap``
    """

    def __init__(self, dataset_json_files, *args, **kwargs):
        """Initialization function detection mean AP (mAP)

        Args:
            dataset_json_files (Dict): paths to the dataset (instance) json files
                for each dataset type and dataset name in the following format:
                ``{'val/detection_coco': '/path/to/instances_val2017.json', ...}``

        """
        super().__init__("detection_mean_ap")
        self.required_params = []
        self.dataset_json_files = dataset_json_files

    def calculate(self, sample_list, model_output, master_only=True, *args, **kwargs):
        """Calculate detection mean AP (mAP) from the prediction list and the dataset
        annotations. The function returns COCO-style mAP@IoU=0.50:0.95.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain
                                "prediction_report" field, which is a list of
                                detection predictions from the model.
            master_only (bool): Whether to only run mAP evaluation on the master node
                                over the gathered detection prediction (to avoid wasting
                                computation and CPU OOM).
                                Default: True (only run mAP evaluation on master).

        Returns:
            torch.FloatTensor: COCO-style mAP@IoU=0.50:0.95.

        """
        # as the detection mAP metric is run on the entire dataset-level predictions,
        # which are *already* gathered from all notes, the evaluation should only happen
        # in one node and broadcasted to other nodes (to avoid CPU OOM due to concurrent
        # mAP evaluation)
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from mmf.utils.distributed import is_master, broadcast_tensor
        from mmf.utils.general import get_current_device

        device = get_current_device()
        if master_only and not is_master():
            # dummy mAP to be override in boardcasting
            mAP = torch.tensor(-1, dtype=torch.float, device=device)
        else:
            predictions = model_output.prediction_report

            cocoGt = COCO(
                self.dataset_json_files[sample_list.dataset_name][
                    sample_list.dataset_type
                ]
            )
            cocoDt = cocoGt.loadRes(predictions)
            cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            mAP = torch.tensor(cocoEval.stats[0], dtype=torch.float, device=device)

        if master_only:
            mAP = broadcast_tensor(mAP, src=0)
        return mAP
