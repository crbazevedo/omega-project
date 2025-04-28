"""
Pacote para o componente ΩTuner.

Este pacote contém a implementação do ΩTuner, responsável por otimizar
hiperparâmetros de modelos e experimentos no sistema Omega.
"""

from .omega_tuner import OmegaTuner, SearchSpace, OptimizationStrategy

__all__ = ['OmegaTuner', 'SearchSpace', 'OptimizationStrategy']
