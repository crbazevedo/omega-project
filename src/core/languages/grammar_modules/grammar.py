"""
Módulo integrador da gramática ΩMetaLang.

Este módulo integra todos os módulos da gramática e constrói a gramática completa
do ΩMetaLang para uso pelo parser.

Autor: Equipe Omega
Data: 28/04/2025
"""

import logging
from . import GRAMMAR_HEADER, COMMON_IMPORTS, GRAMMAR_VERSION, log_grammar_loading
from .types import TYPES_GRAMMAR
from .expressions import EXPRESSIONS_GRAMMAR
from .declarations import DECLARATIONS_GRAMMAR
from .objectives import OBJECTIVES_GRAMMAR
from .program import PROGRAM_GRAMMAR

# Configuração de logging
logger = logging.getLogger(__name__)

def build_complete_grammar():
    """
    Constrói a gramática completa do ΩMetaLang integrando todos os módulos.
    
    Returns:
        String contendo a gramática completa no formato Lark
    """
    logger.info(f"Construindo gramática completa ΩMetaLang v{GRAMMAR_VERSION}")
    
    # Formatar o cabeçalho com a versão atual
    header = GRAMMAR_HEADER.format(version=GRAMMAR_VERSION)
    
    # Concatenar todos os módulos da gramática
    complete_grammar = (
        header +
        PROGRAM_GRAMMAR +
        DECLARATIONS_GRAMMAR +
        TYPES_GRAMMAR +
        EXPRESSIONS_GRAMMAR +
        OBJECTIVES_GRAMMAR +
        COMMON_IMPORTS
    )
    
    logger.debug(f"Gramática completa construída com {len(complete_grammar.splitlines())} linhas")
    return complete_grammar

# Gramática completa do ΩMetaLang
OMEGA_METALANG_GRAMMAR = build_complete_grammar()
