"""
Módulo para a gramática de objetivos do ΩMetaLang.

Este módulo contém as regras de gramática para objetivos e funções de otimização
suportados pela linguagem ΩMetaLang.

Autor: Equipe Omega
Data: 28/04/2025
"""

from . import format_grammar_module, log_grammar_loading

# Registrar carregamento do módulo
log_grammar_loading("objectives")

# Gramática para objetivos
OBJECTIVES_GRAMMAR = format_grammar_module("objectives", r"""
    // Objetivos e funções de otimização
    objective: "Objective" "{" objective_item+ "}"

    objective_item: "Minimize" ":" expression ";"
                  | "Maximize" ":" expression ";"
                  | "Constraint" ":" expression ";"
                  | "WeightS" ":" expression ";"  // λ
                  | "WeightA" ":" expression ";"  // β
                  | "WeightE" ":" expression ";"  // μ
""")
