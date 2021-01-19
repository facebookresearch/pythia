# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.sample import SampleList


class BatchCollator:
    def __init__(self, dataset_name, dataset_type):
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type

    def __call__(self, batch):
        # Create and return sample list with proper name
        # and type set if it is already not a sample list
        # (case of batched iterators)
        sample_list = batch
        if (
            # Check if batch is a list before checking batch[0]
            # or len as sometimes batch is already SampleList
            isinstance(batch, list)
            and len(batch) == 1
            and isinstance(batch[0], SampleList)
        ):
            sample_list = batch[0]
        elif not isinstance(batch, SampleList):
            sample_list = _build_sample_list_with_detr_fields(batch)

        if sample_list._get_tensor_field() is None:
            sample_list = SampleList(sample_list.to_dict())

        sample_list.dataset_name = self._dataset_name
        sample_list.dataset_type = self._dataset_type
        return sample_list


def _build_sample_list_with_detr_fields(batch):
    # handle 'detr_img' and 'detr_target' from detection datasets and
    # collating images of different sizes into a NestedTensor
    from mmf.modules.detr.util.misc import NestedTensor

    detr_img_list = None
    if "detr_img" in batch[0]:
        detr_img_list = [sample.pop("detr_img") for sample in batch]
    detr_target = None
    if "detr_target" in batch[0]:
        detr_target = [sample.pop("detr_target") for sample in batch]
    sample_list = SampleList(batch)
    if detr_img_list is not None:
        sample_list.detr_img = NestedTensor.from_tensor_list(detr_img_list)
    if detr_target is not None:
        sample_list.detr_target = detr_target
    return sample_list
