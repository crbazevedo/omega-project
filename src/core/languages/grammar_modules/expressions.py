"""
Módulo para a gramática de expressões do ΩMetaLang.

Este módulo contém as regras de gramática para expressões, operadores e literais
suportados pela linguagem ΩMetaLang.

Autor: Equipe Omega
Data: 28/04/2025
"""

from . import format_grammar_module, log_grammar_loading

# Registrar carregamento do módulo
log_grammar_loading("expressions")

# Gramática para expressões
EXPRESSIONS_GRAMMAR = format_grammar_module("expressions", r"""
    // Expressões e operadores
    expression: term
              | expression binary_op term -> binary_expression
    
    term: factor
        | unary_op factor -> unary_expression

    factor: literal
          | variable_ref
          | function_call
          | "(" expression ")"
          | list_expr
          | dict_expr

    literal: INTEGER -> integer
           | FLOAT -> float
           | STRING -> string
           | BOOLEAN -> boolean

    INTEGER: /[0-9]+/
    FLOAT: /[0-9]+\.[0-9]*/ | /\.[0-9]+/
    STRING: /"(?:\\.|[^\\"])*"/  // Handle escaped quotes
    BOOLEAN: "true" | "false"

    unary_op: "+" | "-" | "!" | "~"

    binary_op: "+" | "-" | "*" | "/" | "%" | "^"
             | "==" | "!=" | "<" | ">" | "<=" | ">="
             | "&&" | "||" | "&" | "|" | "<<" | ">>"

    function_call: IDENTIFIER "(" [arg_list] ")"

    arg_list: expression ("," expression)*

    list_expr: "[" [expression ("," expression)*] "]"

    dict_expr: "{" [key_value_pair ("," key_value_pair)*] "}"

    key_value_pair: expression ":" expression

    variable_ref: IDENTIFIER | IDENTIFIER "." IDENTIFIER
""")
