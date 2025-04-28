"""
Módulo para o parser da linguagem ΩMetaLang.

Este módulo implementa um parser para a gramática BNF definida em omega_metalang_bnf.md,
utilizando a biblioteca Lark.

Autor: Equipe Omega
Data: 28/04/2025
"""

from lark import Lark, Transformer, v_args
from typing import Dict, Any, List, Optional, Union
import logging
import json
import os

# Configuração de logging
logger = logging.getLogger(__name__)

# Definição da gramática simplificada do ΩMetaLang em formato Lark
# Esta é uma versão inicial baseada na BNF definida em omega_metalang_bnf.md
OMEGA_METALANG_GRAMMAR = r"""
    ?start: omega_program

    omega_program: header declarations objective "}"

    header: "OmegaSpec" IDENTIFIER "{"

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

    objective: "Objective" "{" objective_item+ "}"

    objective_item: "Minimize" ":" expression ";"
                  | "Maximize" ":" expression ";"
                  | "Constraint" ":" expression ";"
                  | "WeightS" ":" expression ";"  // λ
                  | "WeightA" ":" expression ";"  // β
                  | "WeightE" ":" expression ";"  // μ

    type: "Int" | "Float" | "Bool" | "String" 
        | "Vector" "<" type "," expression ">"
        | "Matrix" "<" type "," expression "," expression ">"
        | "Tensor" "<" type "," dimension_list ">"
        | "Distribution" "<" distribution_type ">"
        | "Space" "<" space_type ">"
        | IDENTIFIER  // Tipo definido pelo usuário ou referência

    dimension_list: expression ("," expression)*

    distribution_type: "Normal" | "Uniform" | "Categorical" | "Bernoulli" | IDENTIFIER

    space_type: "Discrete" | "Continuous" | "Box" | IDENTIFIER

    variable_ref: IDENTIFIER | IDENTIFIER "." IDENTIFIER

    expression: literal
              | variable_ref
              | unary_op expression -> unary_expression
              | expression binary_op expression -> binary_expression
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
    STRING: /"[^"]*"/
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

    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/

    %import common.WS
    %import common.NEWLINE
    %ignore WS
    %ignore NEWLINE
    %ignore /\/\/[^\n]*/  // Comentários de linha
    %ignore /\/\*(.|\n)*?\*\//  // Comentários de bloco
"""

class OmegaMetaLangParser:
    """
    Parser para a linguagem ΩMetaLang.
    """
    
    def __init__(self):
        """Inicializa o parser com a gramática definida."""
        try:
            self.parser = Lark(OMEGA_METALANG_GRAMMAR, parser='lalr', transformer=OmegaMetaLangTransformer())
            logger.info("Parser ΩMetaLang inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar o parser ΩMetaLang: {e}")
            raise
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Analisa um texto em ΩMetaLang e retorna uma representação estruturada.
        
        Args:
            text: Código fonte em ΩMetaLang
            
        Returns:
            Representação estruturada do programa
        """
        try:
            result = self.parser.parse(text)
            logger.info("Análise sintática concluída com sucesso")
            return result
        except Exception as e:
            logger.error(f"Erro na análise sintática: {e}")
            raise
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analisa um arquivo em ΩMetaLang e retorna uma representação estruturada.
        
        Args:
            file_path: Caminho para o arquivo
            
        Returns:
            Representação estruturada do programa
        """
        try:
            with open(file_path, 'r') as f:
                text = f.read()
            
            logger.info(f"Arquivo carregado: {file_path}")
            return self.parse(text)
        except Exception as e:
            logger.error(f"Erro ao analisar arquivo {file_path}: {e}")
            raise
    
    def save_ast(self, ast: Dict[str, Any], output_path: str) -> None:
        """
        Salva a AST em formato JSON.
        
        Args:
            ast: Árvore sintática abstrata
            output_path: Caminho para salvar o arquivo JSON
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(ast, f, indent=2)
            
            logger.info(f"AST salva em: {output_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar AST: {e}")
            raise


@v_args(inline=True)
class OmegaMetaLangTransformer(Transformer):
    """
    Transformer para converter a árvore de parse em uma estrutura de dados Python.
    """
    
    def omega_program(self, header, declarations, objective):
        # The closing brace '}' is implicitly handled by Lark and not passed here with inline=True
        return {
            "type": "OmegaProgram",
            "header": header,
            "declarations": declarations,
            "objective": objective
        }
    
    def header(self, name, *args):
        return {
            "type": "Header",
            "name": name
        }
    
    def declarations(self, *declarations):
        return {
            "type": "Declarations",
            "items": list(declarations)
        }
    
    def declaration(self, decl):
        # Pass through the result from the specific declaration type
        return decl
    
    def variable_decl(self, *args):
        # Grammar rules:
        # 1. "Variable" IDENTIFIER ":" type ";" -> args = (name, type_info)
        # 2. "Variable" IDENTIFIER ":" type "=" expression ";" -> args = (name, type_info, expression)
        
        if len(args) == 2:  # Form 1: Variable name : type ;
            name, type_info = args
            return {
                "type": "VariableDeclaration",
                "name": name,
                "var_type": type_info,
                "initial_value": None
            }
        elif len(args) == 3: # Form 2: Variable name : type = expression ;
            name, type_info, expression = args
            return {
                "type": "VariableDeclaration",
                "name": name,
                "var_type": type_info,
                "initial_value": expression
            }
        else:
            # Handle unexpected number of arguments, log error
            logger.error(f"Transformer 'variable_decl' received unexpected number of arguments: {len(args)}, args: {args}")
            # Return a placeholder or raise an error
            return {"type": "VariableDeclaration", "name": "Error", "var_type": "Unknown", "initial_value": None}
    
    def domain_decl(self, name, _, items, __):
        return {
            "type": "DomainDeclaration",
            "name": name,
            "items": items
        }
    
    def domain_item(self, name, _, expression, __):
        return {
            "type": "DomainItem",
            "name": name,
            "value": expression
        }
    
    def model_decl(self, *args):
        # Grammar: "Model" IDENTIFIER "{" model_item+ "}"
        # Expecting name and items, but may receive additional tokens
        if len(args) < 2:
            logger.error(f"Transformer 'model_decl' received too few arguments: {len(args)}")
            return {"type": "ModelDeclaration", "name": "Error", "items": []}
            
        # Extract name (first arg) and items (last arg)
        name = args[0]
        items = args[-1]  # Assuming items is the last argument
        
        return {
            "type": "ModelDeclaration",
            "name": name,
            "items": items
        }
    
    def model_item(self, *args):
        item_type = args[0]
        
        if item_type == "Input":
            return {
                "type": "ModelInput",
                "variable": args[2]
            }
        elif item_type == "Output":
            return {
                "type": "ModelOutput",
                "variable": args[2]
            }
        elif item_type == "Parameter":
            return {
                "type": "ModelParameter",
                "name": args[1],
                "param_type": args[3]
            }
        elif item_type == "Structure":
            return {
                "type": "ModelStructure",
                "structure": args[2]
            }
        elif item_type == "Loss":
            return {
                "type": "ModelLoss",
                "expression": args[2]
            }
    
    def model_structure(self, structure):
        return structure
    
    def predefined_model(self, *args):
        if not args:
            logger.error("Transformer 'predefined_model' received no arguments.")
            return {"type": "PredefinedModel", "model_type": "Unknown"}
        # Assuming the first argument is the token
        model_token = args[0]
        return {
            "type": "PredefinedModel",
            "model_type": str(model_token)
        }
    
    def custom_structure(self, _, layers, __):
        return {
            "type": "CustomStructure",
            "layers": layers
        }
    
    def structure_spec(self, *layers):
        return list(layers)
    
    def layer_spec(self, layer_type, _, params, __, ___):
        return {
            "type": "LayerSpec",
            "layer_type": str(layer_type),
            "params": params
        }
    
    def layer_params(self, *params):
        return list(params)
    
    def param_item(self, name, _, value):
        return {
            "type": "ParameterItem",
            "name": str(name),
            "value": value
        }
    
    def environment_decl(self, name, _, items, __):
        return {
            "type": "EnvironmentDeclaration",
            "name": name,
            "items": items
        }
    
    def environment_item(self, *args):
        item_type = args[0]
        
        if item_type in ["State", "Observation"]:
            return {
                "type": f"Environment{item_type}",
                "variable": args[2]
            }
        elif item_type == "Reward":
            return {
                "type": "EnvironmentReward",
                "expression": args[2]
            }
        elif item_type == "Dynamics":
            return {
                "type": "EnvironmentDynamics",
                "spec": args[2]
            }
        elif item_type == "InitialState":
            return {
                "type": "EnvironmentInitialState",
                "expression": args[2]
            }
    
    def dynamics_spec(self, dynamics_type, _, transition, __):
        return {
            "type": "DynamicsSpec",
            "dynamics_type": str(dynamics_type),
            "transition": transition
        }
    
    def state_transition(self, expression):
        return expression
    
    def api_spec(self, url, *params):
        return {
            "type": "ApiSpec",
            "url": url,
            "params": list(params)
        }
    
    def action_decl(self, name, _, items, __):
        return {
            "type": "ActionDeclaration",
            "name": name,
            "items": items
        }
    
    def action_item(self, *args):
        item_type = args[0]
        
        if item_type == "Type":
            return {
                "type": "ActionType",
                "value": str(args[2])
            }
        elif item_type == "Space":
            return {
                "type": "ActionSpace",
                "variable": args[2]
            }
        elif item_type == "Policy":
            return {
                "type": "ActionPolicy",
                "policy": args[2]
            }
        elif item_type == "Constraints":
            return {
                "type": "ActionConstraints",
                "constraints": args[2]
            }
    
    def policy_spec(self, *args):
        if len(args) == 1:  # Greedy
            return {
                "type": "PolicySpec",
                "policy_type": str(args[0]),
                "params": None
            }
        elif len(args) == 4:  # EpsilonGreedy(expr) or Softmax(expr)
            return {
                "type": "PolicySpec",
                "policy_type": str(args[0]),
                "params": args[2]
            }
        else:  # Custom{expr}
            return {
                "type": "PolicySpec",
                "policy_type": "Custom",
                "params": args[2]
            }
    
    def constraint_list(self, *constraints):
        return list(constraints)
    
    def constraint(self, expression):
        return expression
    
    def objective(self, *args):
        # Grammar: "Objective" "{" objective_item+ "}"
        # Expecting items, but may receive additional tokens
        if len(args) < 1:
            logger.error(f"Transformer 'objective' received too few arguments: {len(args)}")
            return {"type": "Objective", "items": []}
            
        # Assuming items is the first argument after the 'Objective' token, or the last if braces are included
        items = args[0] if isinstance(args[0], list) else args[-1] if isinstance(args[-1], list) else []
        
        return {
            "type": "Objective",
            "items": items
        }
    
    def objective_item(self, *args):
        # Grammar: item_type ":" expression ";"
        # Expecting at least 2 arguments: item_type and expression
        if len(args) < 2:
            logger.error(f"Transformer 'objective_item' received too few arguments: {len(args)}")
            return {"type": "ObjectiveItem", "item_type": "Unknown", "expression": None}
            
        # Extract item_type (first arg) and expression (should be in the middle)
        item_type = args[0]
        # Find expression - typically the second or third argument depending on grammar
        expression = args[1] if len(args) == 2 else args[2] if len(args) >= 3 else None
        
        return {
            "type": f"Objective{item_type}",
            "expression": expression
        }
    
    def type(self, *args):
        if not args: # Handle empty args case
             logger.error("Transformer 'type' method received empty args.")
             # Return a placeholder or raise an error, depending on desired robustness
             return {"type": "Type", "base_type": "Unknown"}

        base_type_token = args[0]
        base_type_str = str(base_type_token)

        if len(args) == 1:  # Tipos básicos ou referências (Int, Float, Bool, String, IDENTIFIER)
            return {
                "type": "Type",
                "base_type": base_type_str
            }
        # Check length before accessing further elements for complex types
        elif base_type_str == "Vector" and len(args) >= 5:  # Vector < type , expression >
            return {
                "type": "Type",
                "base_type": "Vector",
                "parameters": {
                    "element_type": args[2],
                    "size": args[4]
                }
            }
        elif base_type_str == "Matrix" and len(args) >= 7:  # Matrix < type , expression , expression >
            return {
                "type": "Type",
                "base_type": "Matrix",
                "parameters": {
                    "element_type": args[2],
                    "rows": args[4],
                    "cols": args[6]
                }
            }
        elif base_type_str == "Tensor" and len(args) >= 5:  # Tensor < type , dimension_list >
            return {
                "type": "Type",
                "base_type": "Tensor",
                "parameters": {
                    "element_type": args[2],
                    "dimensions": args[4]
                }
            }
        elif base_type_str in ["Distribution", "Space"] and len(args) >= 3:  # Distribution < type > or Space < type >
            return {
                "type": "Type",
                "base_type": base_type_str,
                "parameters": {
                    "subtype": str(args[2]) # Assuming args[2] is the subtype token
                }
            }
        else:
             # Fallback or error for unexpected args structure
             logger.warning(f"Transformer 'type' method received unexpected args structure: {args}")
             # Return a basic type representation based on the first token if possible
             return {
                "type": "Type",
                "base_type": base_type_str
             }

    def dimension_list(self, *dims):
        return list(dims)

    def variable_ref(self, *parts):
        return {
            "type": "VariableReference",
            "name": ".".join(map(str, parts))
        }

    def unary_expression(self, op, expr):
        return {
            "type": "UnaryExpression",
            "operator": str(op),
            "expression": expr
        }

    def binary_expression(self, left, op, right):
        return {
            "type": "BinaryExpression",
            "operator": str(op),
            "left": left,
            "right": right
        }
    
    def function_call(self, name, args=None):
        # Grammar: IDENTIFIER "(" [arg_list] ")"
        # With inline=True, args will be the result of arg_list transformer, or None if arg_list is absent.
        return {
            "type": "FunctionCall",
            "name": str(name),
            "arguments": args if args else []
        }

    def arg_list(self, *args):
        # Returns a list of transformed expressions
        return list(args)

    def list_expr(self, *items):
        # Grammar: "[" [expression ("," expression)*] "]"
        # items will contain the transformed expressions
        return {
            "type": "ListExpression",
            "items": list(items)
        }

    def dict_expr(self, *items):
        # Grammar: "{" [key_value_pair ("," key_value_pair)*] "}"
        # items will contain the transformed key_value_pairs
        return {
            "type": "DictExpression",
            "items": list(items)
        }
    
    def key_value_pair(self, key, _, value):
        return {
            "type": "KeyValuePair",
            "key": key,
            "value": value
        }

    # Métodos para literais
    def integer(self, i):
        return {"type": "Literal", "value": int(i)}

    def float(self, f):
        return {"type": "Literal", "value": float(f)}

    def string(self, s):
        return {"type": "Literal", "value": s[1:-1]} # Remove aspas

    def boolean(self, b):
        return {"type": "Literal", "value": b == "true"}

    # Métodos para tokens (geralmente retornam o próprio token)
    def IDENTIFIER(self, s):
        return str(s)

    def unary_op(self, op):
        return str(op)

    def binary_op(self, op):
        return str(op)

    def action_type(self, t):
        return str(t)

    def distribution_type(self, t):
        return str(t)

    def space_type(self, t):
        return str(t)

    def layer_type(self, t):
        return str(t)

    def dynamics_spec_type(self, t):
        return str(t)

    def policy_spec_type(self, t):
        return str(t)

    def objective_item_type(self, t):
        return str(t)

# Exemplo de uso (para teste rápido)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = OmegaMetaLangParser()
    
    test_code = """
    OmegaSpec SimpleTest {
        Variable x : Int = 10;
        Variable y : Vector<Float, 5>;
        
        Model dummy {
            Input: x;
            Output: y;
            Structure: LinearRegression;
            Loss: MSE(y, predict(x));
        }
        
        Objective {
            Minimize: S;
            WeightS: 1.0;
            WeightA: 0.0;
            WeightE: 0.1;
        }
    }
    """
    
    try:
        ast = parser.parse(test_code)
        print("AST gerada com sucesso:")
        print(json.dumps(ast, indent=2))
        parser.save_ast(ast, "/home/ubuntu/omega_project/examples/ast_example.json")
    except Exception as e:
        print(f"Erro ao analisar o código de teste: {e}")
