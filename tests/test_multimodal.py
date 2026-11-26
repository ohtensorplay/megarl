"""Multimodal trajectories: wire roundtrip and modality-aware collation."""

from megarl.batching import collate_multimodal, from_multimodal
from megarl.contracts import MultimodalTrajectory


def record(request_id=1, image_tokens=(100, 101), response=(11, 12)):
    return MultimodalTrajectory(
        request_id=request_id,
        prompt_len=4,
        token_ids=list(response),
        logprobs=[-0.1] * len(response),
        policy_versions=[7] * len(response),
        image_token_ids=list(image_tokens),
        image_insert_at=2,
        image_grid=(2, 1),
        modality_mask=[0, 0] + [1] * len(image_tokens) + [0] * len(response),
        reward=1.0,
        finish_reason=0,
        payload_handle=None,
    )


def test_image_tokens_are_spliced_into_prompt_with_mask():
    sample = from_multimodal(record(), [1, 2, 3, 4])
    ids, modality, loss = sample.compose()
    assert ids == [1, 2, 100, 101, 3, 4, 11, 12]
    assert modality == [0, 0, 1, 1, 0, 0, 0, 0]
    assert loss == [0] * 6 + [1, 1]


def test_collate_multimodal_pads_and_aligns_logprobs():
    first = from_multimodal(record(1), [1, 2, 3, 4])
    second = from_multimodal(record(2, (100,), (21,)), [5, 6, 7, 8])
    batch = collate_multimodal([first, second])
    assert batch.max_len == 8
    assert batch.cu_seqlens == [0, 8, 14]
    assert batch.modality_mask[1] == [0, 0, 1, 0, 0, 0, 0, 0]
    assert batch.loss_mask[1] == [0] * 5 + [1] + [0] * 2
    assert batch.old_logprobs[1] == [0.0] * 5 + [-0.1] + [0.0] * 2
    assert batch.versions[1] == [0] * 5 + [7] + [0] * 2
