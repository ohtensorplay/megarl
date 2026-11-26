"""Structural checks for the namespaced dLLM algorithm packages.

The reference RL projects (ROLL ``pipeline/diffusion/<algo>``, trl
``experimental/<algo>``, AReaL ``trainer/<algo>``) organise each algorithm as a
subpackage rather than a flat module. megarl follows that convention: there are
no flat ``dllm_<algo>.py`` modules and no compatibility shims.
"""

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "megarl"

ALGORITHMS = {
    "agrpo": {
        "arxiv": "2510.04019",
        "modules": {"config", "types", "rollout", "batching", "trainer", "controller"},
    },
    "continuous": {
        "arxiv": "2505.05470",
        "modules": {"config", "types", "batching", "loss", "rollout", "trainer", "controller"},
    },
    "dopsd": {
        "arxiv": "2607.04428",
        "modules": {"config", "types", "rollout", "batching", "loss", "trainer", "controller"},
    },
    "gdsd": {
        "arxiv": "2605.29398",
        "modules": {"config", "types", "rollout", "batching", "loss", "trainer", "controller"},
    },
    "justgrpo": {
        "arxiv": "2601.15165",
        "modules": {"config", "types", "rollout", "batching", "trainer", "controller"},
    },
    "process": {
        "arxiv": "2509.06949",
        "modules": {"config", "value", "gae", "trainer"},
    },
    "purl": {
        "arxiv": "2611.11551v1",
        "modules": {
            "config", "types", "model", "rollout", "batching", "loss",
            "trainer", "controller",
        },
    },
    "remarl": {
        "arxiv": "2611.22576",
        "modules": {
            "config", "types", "policy", "rollout", "batching", "loss",
            "trainer", "controller",
        },
    },
    "t3d": {
        "arxiv": "2602.12262",
        "modules": {"config", "types", "rollout", "batching", "loss", "trainer", "controller"},
    },
    "tips": {
        "arxiv": "2610.22099",
        "modules": {
            "config", "types", "policy", "rollout", "batching", "loss",
            "trainer", "controller",
        },
    },
}


def _package(algo):
    return importlib.import_module(f"megarl.dllm.algorithms.{algo}")


@pytest.mark.parametrize("algo,spec", ALGORITHMS.items())
def test_algorithm_subpackage_exports_resolve(algo, spec):
    package = _package(algo)
    assert package.__all__
    for name in package.__all__:
        assert getattr(package, name, None) is not None, (algo, name)


@pytest.mark.parametrize("algo,spec", ALGORITHMS.items())
def test_algorithm_subpackage_is_split_by_responsibility(algo, spec):
    package = _package(algo)
    modules = {p.stem for p in Path(package.__path__[0]).glob("*.py") if p.name != "__init__.py"}
    assert spec["modules"] <= modules


@pytest.mark.parametrize("algo,spec", ALGORITHMS.items())
def test_each_module_cites_the_paper(algo, spec):
    package = _package(algo)
    for path in Path(package.__path__[0]).glob("*.py"):
        doc = path.read_text().lower()
        assert "arxiv" in doc, path
        assert spec["arxiv"] in doc, path


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_no_flat_module_or_shim(algo):
    assert not (PACKAGE_ROOT / f"dllm_{algo}.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"megarl.dllm_{algo}")


def test_family_namespace_exposes_every_algorithm():
    family = importlib.import_module("megarl.dllm.algorithms")
    assert family.__all__ == sorted(ALGORITHMS)
    for algo in ALGORITHMS:
        assert getattr(family, algo) is _package(algo)


def test_top_level_exports_come_from_namespaced_packages():
    import megarl

    for algo in ALGORITHMS:
        canonical = _package(algo)
        for name in canonical.__all__:
            assert getattr(megarl, name) is getattr(canonical, name)


def test_base_dllm_api_is_also_namespaced():
    family = importlib.import_module("megarl.dllm")
    assert {"config", "adapters", "estimator", "batching", "trainer", "controller",
            "rollout", "trajectory", "algorithms"} <= {
                p.stem for p in Path(family.__path__[0]).iterdir() if p.is_dir()
            } | {p.stem for p in Path(family.__path__[0]).glob("*.py")}
    assert not (PACKAGE_ROOT / "dllm.py").exists()
    assert not (PACKAGE_ROOT / "dllm_rollout.py").exists()
    assert not (PACKAGE_ROOT / "dllm_trajectory.py").exists()
