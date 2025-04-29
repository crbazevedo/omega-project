"""
Módulo para a gramática de declarações do ΩMetaLang.

Este módulo contém as regras de gramática para todas as declarações suportadas 
pela linguagem ΩMetaLang, incluindo variáveis, modelos, ambientes e ações.

Autor: Equipe Omega
Data: 28/04/2025
"""

from . import format_grammar_module, log_grammar_loading

# Registrar carregamento do módulo
log_grammar_loading("declarations")

# Gramática para declarações
DECLARATIONS_GRAMMAR = format_grammar_module("declarations", r"""
    // Declarações
    declarations: declaration*

    declaration: variable_decl
               | domain_decl
               | model_decl
               | environment_decl
               | action_decl

    variable_decl: "Variable" IDENTIFIER ":" type ";" 
                 | "Variable" IDENTIFIER ":" type "=" expression ";"

    domain_decl: "Domain" IDENTIFIER "{" domain_item+ "}"

    domain_item: IDENTIFIER ":" expression ";"

    model_decl: "Model" IDENTIFIER "{" model_item+ "}"

    model_item: "Input" ":" variable_ref ";"
              | "Output" ":" variable_ref ";"
              | "Parameter" IDENTIFIER ":" type ";"
              | "Structure" ":" model_structure ";"
              | "Loss" ":" expression ";"

    model_structure: predefined_model
                   | custom_structure

    predefined_model: "LinearRegression" 
                    | "LogisticRegression" 
                    | "NeuralNetwork" 
                    | "DecisionTree" 
                    | "RandomForest" 
                    | "GaussianProcess"
                    | IDENTIFIER  // Referência a um modelo na ΩIntelligenceBase

    custom_structure: "Custom" "{" structure_spec "}"

    structure_spec: layer_spec+

    layer_spec: layer_type "(" layer_params ")" ";"

    layer_type: "Dense" | "Conv2D" | "LSTM" | "Attention" | IDENTIFIER

    layer_params: param_item ("," param_item)*

    param_item: IDENTIFIER "=" expression

    environment_decl: "Environment" IDENTIFIER "{" environment_item+ "}"

    environment_item: "State" ":" variable_ref ";"
                    | "Observation" ":" variable_ref ";"
                    | "Reward" ":" expression ";"
                    | "Dynamics" ":" dynamics_spec ";"
                    | "InitialState" ":" expression ";"

    dynamics_spec: "Deterministic" "{" state_transition "}"
                 | "Stochastic" "{" state_transition "}"
                 | "External" "{" api_spec "}"

    state_transition: expression

    api_spec: STRING ("," param_item)*

    action_decl: "Action" IDENTIFIER "{" action_item+ "}"

    action_item: "Type" ":" action_type ";"
               | "Space" ":" variable_ref ";"
               | "Policy" ":" policy_spec ";"
               | "Constraints" ":" constraint_list ";"

    action_type: "Discrete" | "Continuous" | "Mixed"

    policy_spec: "Greedy" 
               | "EpsilonGreedy" "(" expression ")"
               | "Softmax" "(" expression ")"
               | "Custom" "{" expression "}"

    constraint_list: constraint ("," constraint)*

    constraint: expression
""")
