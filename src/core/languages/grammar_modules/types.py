"""
Módulo para a gramática de tipos do ΩMetaLang.

Este módulo contém as regras de gramática para todos os tipos suportados pela linguagem ΩMetaLang.

Autor: Equipe Omega
Data: 28/04/2025
"""

from . import format_grammar_module, log_grammar_loading

# Registrar carregamento do módulo
log_grammar_loading("types")

# Gramática para tipos
TYPES_GRAMMAR = format_grammar_module("types", r"""
    // Definição de tipos
    type: simple_type
        | vector_type
        | matrix_type
        | tensor_type
        | distribution_type
        | space_type
        | predefined_distribution_type
        | predefined_space_type
        | generic_type
        | custom_type

    simple_type: "Int" | "Float" | "Bool" | "String"
    
    // Tipos genéricos e parametrizados
    vector_type: "Vector" "<" type ("," expression)? ">"
    matrix_type: "Matrix" "<" type ("," expression "," expression)? ">"
    tensor_type: "Tensor" "<" type ("," dimension_list)? ">"
    distribution_type: "Distribution" "<" type ">"
    space_type: "Space" "<" type ">"
    
    // Tipos de distribuição e espaço predefinidos
    predefined_distribution_type: "Distribution" "<" distribution_type_name ">"
    predefined_space_type: "Space" "<" space_type_name ">"
    
    // Novo: Suporte para tipos genéricos
    generic_type: IDENTIFIER "<" type_param_list ">"
    type_param_list: type ("," type)*
    
    custom_type: IDENTIFIER  // Tipo definido pelo usuário ou referência

    dimension_list: expression ("," expression)*

    distribution_type_name: "Normal" | "Uniform" | "Categorical" | "Bernoulli"
    space_type_name: "Discrete" | "Continuous" | "Box"
""")
