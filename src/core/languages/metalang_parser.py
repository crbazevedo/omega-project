"""
Módulo para o parser da linguagem ΩMetaLang.

Este módulo implementa um parser para a gramática BNF definida em omega_metalang_bnf.md,
utilizando a biblioteca Lark.

Autor: Equipe Omega
Data: 28/04/2025
"""

from lark import Lark, Transformer, v_args, Token
from typing import Dict, Any, List, Optional, Union
import logging
import json
import os

# Configuração de logging
logger = logging.getLogger(__name__)

# Definição da gramática do ΩMetaLang em formato Lark
# Versão revisada para melhor tratamento de tipos
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

    // Revisão da regra 'type' para melhor transformação
    type: simple_type
        | vector_type
        | matrix_type
        | tensor_type
        | distribution_type
        | space_type
        | custom_type

    simple_type: "Int" | "Float" | "Bool" | "String"
    vector_type: "Vector" "<" type "," expression ">"
    matrix_type: "Matrix" "<" type "," expression "," expression ">"
    tensor_type: "Tensor" "<" type "," dimension_list ">"
    distribution_type: "Distribution" "<" distribution_type_name ">"
    space_type: "Space" "<" space_type_name ">"
    custom_type: IDENTIFIER  // Tipo definido pelo usuário ou referência

    dimension_list: expression ("," expression)*

    distribution_type_name: "Normal" | "Uniform" | "Categorical" | "Bernoulli" | IDENTIFIER
    space_type_name: "Discrete" | "Continuous" | "Box" | IDENTIFIER

    variable_ref: IDENTIFIER | IDENTIFIER "." IDENTIFIER

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
            # Usar LALR para eficiência, mas pode exigir ajustes na gramática se houver conflitos
            self.parser = Lark(OMEGA_METALANG_GRAMMAR, parser='lalr', transformer=OmegaMetaLangTransformer())
            logger.info("Parser ΩMetaLang inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar o parser ΩMetaLang: {e}")
            raise
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Analisa um texto em ΩMetaLang e retorna uma representação estruturada (AST).
        
        Args:
            text: Código fonte em ΩMetaLang
            
        Returns:
            AST como um dicionário Python
        """
        try:
            result = self.parser.parse(text)
            logger.info("Análise sintática concluída com sucesso")
            return result
        except Exception as e:
            logger.error(f"Erro na análise sintática: {e}")
            # Considerar retornar uma AST de erro ou informações mais detalhadas
            raise
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analisa um arquivo em ΩMetaLang e retorna a AST.
        
        Args:
            file_path: Caminho para o arquivo
            
        Returns:
            AST como um dicionário Python
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


# Usar @v_args(inline=True) seletivamente para simplificar métodos
class OmegaMetaLangTransformer(Transformer):
    """
    Transformer para converter a árvore de parse em uma AST Python.
    """
    
    # --- Métodos para Literais --- 
    @v_args(inline=True)
    def integer(self, value):
        return {"type": "Literal", "value": int(value), "literal_type": "Int"}
    
    @v_args(inline=True)
    def float(self, value):
        return {"type": "Literal", "value": float(value), "literal_type": "Float"}
    
    @v_args(inline=True)
    def string(self, value):
        # Remover as aspas externas
        return {"type": "Literal", "value": value[1:-1], "literal_type": "String"}
    
    @v_args(inline=True)
    def boolean(self, value):
        return {"type": "Literal", "value": value == "true", "literal_type": "Bool"}

    # --- Métodos para Estrutura Principal --- 
    @v_args(inline=True)
    def omega_program(self, header, declarations, objective):
        return {
            "type": "OmegaProgram",
            "header": header,
            "declarations": declarations,
            "objective": objective
        }
    
    @v_args(inline=True)
    def header(self, name):
        return {
            "type": "Header",
            "name": str(name) # Convert Token to string
        }
    
    def declarations(self, *declarations):
        return {
            "type": "Declarations",
            "items": list(declarations)
        }
    
    # Pass-through para declarações específicas
    def declaration(self, decl):
        return decl[0] # Unwrap the list if Lark wraps single items

    # --- Métodos para Declarações --- 
    @v_args(inline=True)
    def variable_decl(self, name, type_info, initial_value=None):
        # Assumes Lark passes None for optional parts like `= expression`
        return {
            "type": "VariableDeclaration",
            "name": str(name),
            "var_type": type_info, # Should be the dict from the 'type' method
            "initial_value": initial_value
        }

    @v_args(inline=True)
    def domain_decl(self, name, *items):
        return {
            "type": "DomainDeclaration",
            "name": str(name),
            "items": list(items)
        }
    
    @v_args(inline=True)
    def domain_item(self, name, value):
        return {
            "type": "DomainItem",
            "name": str(name),
            "value": value
        }
    
    @v_args(inline=True)
    def model_decl(self, name, *items):
        return {
            "type": "ModelDeclaration",
            "name": str(name),
            "items": list(items)
        }
    
    # Model items require specific handling based on keyword
    def model_item(self, tree):
        item_type_token = tree.children[0]
        item_type = str(item_type_token)
        
        if item_type == "Input":
            return {"type": "ModelInput", "variable": tree.children[2]}
        elif item_type == "Output":
            return {"type": "ModelOutput", "variable": tree.children[2]}
        elif item_type == "Parameter":
            return {"type": "ModelParameter", "name": str(tree.children[1]), "param_type": tree.children[3]}
        elif item_type == "Structure":
            return {"type": "ModelStructure", "structure": tree.children[2]}
        elif item_type == "Loss":
            return {"type": "ModelLoss", "expression": tree.children[2]}
        else:
            logger.error(f"Unknown model item type: {item_type}")
            return {"type": "Error", "message": f"Unknown model item {item_type}"}

    def model_structure(self, structure):
        return structure[0]

    @v_args(inline=True)
    def predefined_model(self, model_type):
        return {
            "type": "PredefinedModel",
            "model_type": str(model_type)
        }
    
    @v_args(inline=True)
    def custom_structure(self, layers):
        return {
            "type": "CustomStructure",
            "layers": layers
        }
    
    def structure_spec(self, *layers):
        return list(layers)
    
    @v_args(inline=True)
    def layer_spec(self, layer_type, params):
        return {
            "type": "LayerSpec",
            "layer_type": str(layer_type),
            "params": params
        }
    
    def layer_params(self, *params):
        return list(params)
    
    @v_args(inline=True)
    def param_item(self, name, value):
        return {
            "type": "ParameterItem",
            "name": str(name),
            "value": value
        }

    @v_args(inline=True)
    def environment_decl(self, name, *items):
        return {
            "type": "EnvironmentDeclaration",
            "name": str(name),
            "items": list(items)
        }

    # Environment items require specific handling
    def environment_item(self, tree):
        item_type = str(tree.children[0])
        if item_type in ["State", "Observation"]:
            return {"type": f"Environment{item_type}", "variable": tree.children[2]}
        elif item_type == "Reward":
            return {"type": "EnvironmentReward", "expression": tree.children[2]}
        elif item_type == "Dynamics":
            return {"type": "EnvironmentDynamics", "spec": tree.children[2]}
        elif item_type == "InitialState":
            return {"type": "EnvironmentInitialState", "expression": tree.children[2]}
        else:
            logger.error(f"Unknown environment item type: {item_type}")
            return {"type": "Error", "message": f"Unknown environment item {item_type}"}

    # Dynamics spec requires specific handling
    def dynamics_spec(self, tree):
        spec_type = str(tree.children[0])
        if spec_type in ["Deterministic", "Stochastic"]:
            return {"type": "DynamicsSpec", "dynamics_type": spec_type, "transition": tree.children[2]}
        elif spec_type == "External":
            # Children: "External", "{", api_spec, "}"
            return {"type": "DynamicsSpec", "dynamics_type": "External", "api": tree.children[2]}
        else:
             logger.error(f"Unknown dynamics spec type: {spec_type}")
             return {"type": "Error", "message": f"Unknown dynamics spec {spec_type}"}

    def state_transition(self, expression):
        return expression[0]

    @v_args(inline=True)
    def api_spec(self, url, *params):
        return {
            "type": "ApiSpec",
            "url": url, # Should be a string literal result
            "params": list(params)
        }

    @v_args(inline=True)
    def action_decl(self, name, *items):
        return {
            "type": "ActionDeclaration",
            "name": str(name),
            "items": list(items)
        }

    # Action items require specific handling
    def action_item(self, tree):
        item_type = str(tree.children[0])
        if item_type == "Type":
            return {"type": "ActionType", "value": str(tree.children[2])}
        elif item_type == "Space":
            return {"type": "ActionSpace", "variable": tree.children[2]}
        elif item_type == "Policy":
            return {"type": "ActionPolicy", "policy": tree.children[2]}
        elif item_type == "Constraints":
            return {"type": "ActionConstraints", "constraints": tree.children[2]}
        else:
            logger.error(f"Unknown action item type: {item_type}")
            return {"type": "Error", "message": f"Unknown action item {item_type}"}

    @v_args(inline=True)
    def action_type(self, type_name):
        return str(type_name)

    # Policy spec requires specific handling
    def policy_spec(self, tree):
        policy_type = str(tree.children[0])
        if policy_type == "Greedy":
            return {"type": "PolicySpec", "policy_type": "Greedy", "params": None}
        elif policy_type in ["EpsilonGreedy", "Softmax"]:
            # Children: type_name, "(", expression, ")"
            return {"type": "PolicySpec", "policy_type": policy_type, "params": tree.children[2]}
        elif policy_type == "Custom":
            # Children: "Custom", "{", expression, "}"
            return {"type": "PolicySpec", "policy_type": "Custom", "params": tree.children[2]}
        else:
            logger.error(f"Unknown policy spec type: {policy_type}")
            return {"type": "Error", "message": f"Unknown policy spec {policy_type}"}

    def constraint_list(self, *constraints):
        return list(constraints)

    def constraint(self, expression):
        return expression[0]

    @v_args(inline=True)
    def objective(self, *items):
        return {
            "type": "Objective",
            "items": list(items)
        }

    # Objective items require specific handling
    def objective_item(self, tree):
        item_type = str(tree.children[0])
        if item_type in ["Minimize", "Maximize", "Constraint", "WeightS", "WeightA", "WeightE"]:
            return {"type": f"Objective{item_type}", "expression": tree.children[2]}
        else:
            logger.error(f"Unknown objective item type: {item_type}")
            return {"type": "Error", "message": f"Unknown objective item {item_type}"}

    # --- Métodos para Tipos (Revisado - Strategy 3) ---
    @v_args(inline=True)
    def simple_type(self, type_name):
        # Retorna a estrutura de dicionário final diretamente
        return {"type": "Type", "base_type": str(type_name)}

    @v_args(inline=True)
    def vector_type(self, element_type_dict, size_expr_dict):
        # element_type_dict é o resultado da regra interna 'type'
        # size_expr_dict é o resultado da regra 'expression'
        return {"type": "Type", "base_type": "Vector", "element_type": element_type_dict, "size": size_expr_dict}

    @v_args(inline=True)
    def matrix_type(self, element_type_dict, rows_expr_dict, cols_expr_dict):
        return {"type": "Type", "base_type": "Matrix", "element_type": element_type_dict, "rows": rows_expr_dict, "cols": cols_expr_dict}

    @v_args(inline=True)
    def tensor_type(self, element_type_dict, dimensions_list):
        return {"type": "Type", "base_type": "Tensor", "element_type": element_type_dict, "dimensions": dimensions_list}

    @v_args(inline=True)
    def distribution_type(self, dist_name_str):
        return {"type": "Type", "base_type": "Distribution", "distribution": dist_name_str}

    @v_args(inline=True)
    def space_type(self, space_name_str):
        return {"type": "Type", "base_type": "Space", "space": space_name_str}

    @v_args(inline=True)
    def custom_type(self, type_name):
        # Retorna a estrutura de dicionário final diretamente
        return {"type": "Type", "base_type": str(type_name), "is_custom": True}

    # Métodos auxiliares permanecem os mesmos
    @v_args(inline=True)
    def distribution_type_name(self, name):
        return str(name)

    @v_args(inline=True)
    def space_type_name(self, name):
        return str(name)

    def dimension_list(self, *dims):
         return list(dims)

    # O método 'def type(self, tree):' foi REMOVIDO

    # --- Métodos para Expressões --- 
    @v_args(inline=True)
    def variable_ref(self, name1, name2=None):
        if name2:
            return {"type": "VariableRef", "object": str(name1), "attribute": str(name2)}
        else:
            return {"type": "VariableRef", "name": str(name1)}

    def unary_expression(self, op, operand):
        return {"type": "UnaryExpression", "operator": str(op), "operand": operand}

    def binary_expression(self, left, op, right):
        return {"type": "BinaryExpression", "left": left, "operator": str(op), "right": right}

    @v_args(inline=True)
    def function_call(self, name, args=None):
        return {
            "type": "FunctionCall",
            "name": str(name),
            "arguments": args if args else []
        }
    
    # Handle arg_list potentially being empty or single
    def arg_list(self, *args):
        return list(args)

    def list_expr(self, *items):
        return {"type": "ListExpression", "items": list(items)}

    def dict_expr(self, *items):
        return {"type": "DictExpression", "items": list(items)}

    @v_args(inline=True)
    def key_value_pair(self, key, value):
        return {"key": key, "value": value}

    # --- Métodos para Terminais --- 
    # Convert Lark Tokens to strings where appropriate (e.g., IDENTIFIER)
    def IDENTIFIER(self, token):
        return str(token)

    # Default behavior for terminals if not handled above
    def __default__(self, data, children, meta):
        # Fallback for rules not explicitly handled
        # logger.warning(f"Using default transformer for rule: {data}")
        if len(children) == 1:
            return children[0] # Pass through single child
        return children # Return list of children

# Exemplo de uso (para teste rápido)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = OmegaMetaLangParser()
    
    test_code = """
    OmegaSpec SimpleTest {
        Variable x : Int = 10;
        Variable y : Vector<Float, 5>;
        Variable m : Matrix<Int, 2, 3>;
        Variable d : Distribution<Normal>;
        Variable s : Space<Discrete>;
        Variable ct : CustomType;
        
        Objective {
            Minimize: S;
        }
    }
    """
    
    try:
        ast = parser.parse(test_code)
        print(json.dumps(ast, indent=2))
    except Exception as e:
        print(f"Erro: {e}")

    test_code_2 = """
    OmegaSpec RegressionTest {
        Variable input_data : Matrix<Float, 100, 10>;
        Variable target_data : Vector<Float, 100>;
        
        Model linear_reg {
            Input: input_data;
            Output: target_data;
            Structure: LinearRegression;
            Loss: MeanSquaredError(output, target_data);
        }
        
        Objective {
            Minimize: C + E;
            WeightS: 0.0;
            WeightA: 1.0;
            WeightE: 0.01;
        }
    }
    """
    try:
        ast_2 = parser.parse(test_code_2)
        print("\n--- AST 2 ---")
        print(json.dumps(ast_2, indent=2))
    except Exception as e:
        print(f"Erro no segundo teste: {e}")

