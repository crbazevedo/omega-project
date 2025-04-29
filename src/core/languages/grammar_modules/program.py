"""
Módulo para a gramática principal do ΩMetaLang.

Este módulo contém as regras de gramática para a estrutura principal do programa
e integra todos os outros módulos da gramática.

Autor: Equipe Omega
Data: 28/04/2025
"""

from . import format_grammar_module, log_grammar_loading, GRAMMAR_HEADER, COMMON_IMPORTS

# Registrar carregamento do módulo
log_grammar_loading("program")

# Gramática para a estrutura principal do programa
PROGRAM_GRAMMAR = format_grammar_module("program", r"""
    // Estrutura principal do programa
    omega_program: header declarations? objective? "}"

    header: "OmegaSpec" IDENTIFIER "{"

    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
""")
