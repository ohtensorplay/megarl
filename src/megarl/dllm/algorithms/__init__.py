"""dLLM reinforcement-learning algorithm recipes.

One subpackage per algorithm, mirroring ROLL
``pipeline/diffusion/<algo>``, trl ``experimental/<algo>`` and AReaL
``trainer/<algo>``. Import from ``megarl.dllm.algorithms.<algorithm>``.
"""

from . import agrpo, continuous, dopsd, gdsd, justgrpo, process, purl, remarl, t3d, tips

__all__ = ['agrpo', 'continuous', 'dopsd', 'gdsd', 'justgrpo', 'process', 'purl', 'remarl', 't3d', 'tips']
