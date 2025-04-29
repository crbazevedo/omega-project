"""
Módulo para a gramática de importações e módulos do ΩMetaLang.

Este módulo contém as regras de gramática para importações e módulos
suportados pela linguagem ΩMetaLang.

Autor: Equipe Omega
Data: 28/04/2025
"""

from . import format_grammar_module, log_grammar_loading

# Registrar carregamento do módulo
log_grammar_loading("imports")

# Gramática para importações e módulos
IMPORTS_GRAMMAR = format_grammar_module("imports", r"""
    // Importações e módulos
    import_statement: "Import" import_target ";"
    
    import_target: module_path
                 | module_path "." "{" import_list "}"
    
    module_path: IDENTIFIER ("." IDENTIFIER)*
    
    import_list: IDENTIFIER ("," IDENTIFIER)*
""")
