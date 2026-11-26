"""Versioned LoRA adapter lifecycle: publish, activate and stale GC."""

import os

import pytest
import tensorplay as tp

from megarl.adapters import AdapterRegistry, VersionedAdapterRollout
from megarl.weights import DigestMismatch


def state(value: float):
    return {"w": tp.tensor([value, value + 1.0])}


def test_publish_loads_before_version_bump_and_keeps_staleness_window(tmp_path):
    events = []

    def load(artifact):
        events.append(("load", artifact.version, registry.active_version))

    def unload(artifact):
        events.append(("unload", artifact.version, registry.active_version))

    registry = AdapterRegistry(tmp_path, name="policy", max_staleness=1, load=load, unload=unload)
    for version in range(1, 6):
        artifact = registry.reload(state(float(version)))
        assert artifact.version == version
        assert registry.active_version == version

    # The adapter exists on the engine before the version moves, and the
    # version at load time is always the previous one.
    loads = [event for event in events if event[0] == "load"]
    assert loads == [
        ("load", version, version - 1) for version in range(1, 6)
    ]
    assert [event[1] for event in events if event[0] == "unload"] == [1, 2, 3]

    status = registry.status()
    assert status["active_version"] == 5
    assert status["servable"] == [4, 5]
    assert sorted(os.listdir(tmp_path)) == [
        "manifest.json",
        "policy-v3",
        "policy-v4",
        "policy-v5",
    ]
    assert registry.latest_version == 5


def test_registry_resumes_monotonic_versions_and_rejects_tampering(tmp_path):
    registry = AdapterRegistry(tmp_path, max_staleness=2)
    registry.reload(state(1.0))
    registry.reload(state(2.0))
    restored = AdapterRegistry(tmp_path, max_staleness=2)
    assert restored.active_version == 2
    assert restored.latest_version == 2
    assert restored.reload(state(3.0)).version == 3

    artifact = restored.adapters[3]
    with open(os.path.join(artifact.path, "adapter.mega"), "ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(DigestMismatch):
        AdapterRegistry(tmp_path, max_staleness=2)


def test_max_adapters_bound_never_evicts_active(tmp_path):
    registry = AdapterRegistry(tmp_path, max_staleness=1, max_adapters=2)
    for version in range(1, 5):
        registry.reload(state(float(version)))
    assert registry.active_version == 4
    assert registry.adapters[4].version == 4
    assert len(registry.adapters) <= 2  # capacity bound, never the active one


def test_versioned_rollout_wrapper_forwards_generate(tmp_path):
    class Inner:
        def generate(self, request):
            return [request]

        def reload_weights(self, path, expected_digest=None):
            raise AssertionError("must be intercepted")

    class Request:
        pass

    inner = Inner()
    registry = AdapterRegistry(tmp_path)
    wrapper = VersionedAdapterRollout(inner, registry)
    request = Request()
    assert wrapper.generate(request) == [request]
    wrapper.reload_weights(state(1.0))
    assert registry.active_version == 1
    assert registry.status()["servable"] == [1]
    with pytest.raises(DigestMismatch):
        wrapper.reload_weights(state(2.0), expected_digest="deadbeef")
