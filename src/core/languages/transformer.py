from lark import Transformer, v_args, Token, Tree
import logging

# Configuração de logging
logger = logging.getLogger(__name__)

class OmegaMetaLangTransformer(Transformer):
    """
    Transformer para converter a árvore de parse do ΩMetaLang em uma AST estruturada.
    
    Esta classe define métodos para transformar cada regra da gramática em
    uma representação mais útil para processamento posterior.
    """
    
    # --- Métodos para o Programa ---
    @v_args(inline=True)
    def program(self, header, declarations=None, objective=None):
        result = {
            "header": header,
        }
        
        if declarations:
            result["declarations"] = declarations
        else:
            result["declarations"] = {"type": "Declarations", "items": []}
            
        if objective:
            result["objective"] = objective
        else:
            result["objective"] = {"type": "Objective", "items": []}
            
        return result
    
    @v_args(inline=True)
    def header(self, name):
        return {"type": "Header", "name": str(name)}
    
    def declarations(self, items):
        return {"type": "Declarations", "items": items}
    
    @v_args(inline=True)
    def objective(self, items):
        try:
            return {"type": "Objective", "items": items}
        except Exception as e:
            logger.error(f"Unknown objective item type: {items}")
            return {"type": "Objective", "items": []}
    
    def objective_items(self, items):
        return items
    
    @v_args(inline=True)
    def objective_item(self, objective_type, expression=None):
        if expression is None:
            return {"objective_type": str(objective_type)}
        return {"objective_type": str(objective_type), "expression": expression}
    
    # --- Métodos para Declarações ---
    @v_args(inline=True)
    def variable_declaration(self, name, var_type, initial_value=None):
        result = Tree("VariableDeclaration", [name, var_type])
        if initial_value is not None:
            result.children.append(initial_value)
        return result
    
    @v_args(inline=True)
    def model_declaration(self, name, model_type=None, model_body=None):
        result = {"type": "ModelDeclaration", "name": str(name)}
        if model_type:
            result["model_type"] = model_type
        if model_body:
            result["body"] = model_body
        return result
    
    @v_args(inline=True)
    def environment_declaration(self, name, env_type=None, env_body=None):
        result = {"type": "EnvironmentDeclaration", "name": str(name)}
        if env_type:
            result["env_type"] = env_type
        if env_body:
            result["body"] = env_body
        return result
    
    @v_args(inline=True)
    def action_declaration(self, name, action_type=None, action_body=None):
        result = {"type": "ActionDeclaration", "name": str(name)}
        if action_type:
            result["action_type"] = action_type
        if action_body:
            result["body"] = action_body
        return result
    
    @v_args(inline=True)
    def domain_declaration(self, name, domain_type=None, domain_body=None):
        result = {"type": "DomainDeclaration", "name": str(name)}
        if domain_type:
            result["domain_type"] = domain_type
        if domain_body:
            result["body"] = domain_body
        return result
    
    def model_body(self, items):
        return {"type": "ModelBody", "items": items}
    
    def environment_body(self, items):
        return {"type": "EnvironmentBody", "items": items}
    
    def action_body(self, items):
        return {"type": "ActionBody", "items": items}
    
    def domain_body(self, items):
        return {"type": "DomainBody", "items": items}
    
    # --- Métodos para Tipos ---
    def type(self, children):
        # A regra 'type' é uma escolha, então o filho é o resultado do tipo específico
        return children[0]
    
    # @v_args(inline=True)  # Removido para depuração
    def simple_type(self, children):
        # Depurar o que está sendo recebido
        logger.debug(f"simple_type received: {children}")
        
        # Se não houver filhos, tentar extrair o tipo do contexto
        if not children or len(children) == 0:
            # Verificar se temos acesso ao contexto atual
            if hasattr(self, '_tokentype') and self._tokentype:
                type_str = self._tokentype
            else:
                # Tentar extrair do contexto da pilha de chamadas
                import inspect
                frame = inspect.currentframe()
                try:
                    # Verificar frames anteriores para encontrar informações sobre o tipo
                    while frame:
                        if 'token' in frame.f_locals and hasattr(frame.f_locals['token'], 'type'):
                            token_type = frame.f_locals['token'].type
                            if token_type in ["INT", "FLOAT", "BOOL", "STRING"]:
                                type_str = token_type.capitalize()
                                break
                        frame = frame.f_back
                    else:
                        # Se não encontrar, usar o tipo do contexto atual
                        type_str = "Unknown"
                finally:
                    del frame  # Evitar referências circulares
                
            logger.debug(f"Extracted type from context: {type_str}")
            return Tree("SimpleType", [Token("IDENTIFIER", type_str)])
            
        # Extrair o tipo do primeiro filho
        type_token = children[0]
        if isinstance(type_token, Token):
            type_str = type_token.value
        else:
            type_str = str(type_token)
            
        logger.debug(f"Extracted type from children: {type_str}")
        
        # Verificar se é um tipo primitivo conhecido
        if type_str in ["Int", "Float", "Bool", "String", "Double"]:
            return Tree("SimpleType", [Token("IDENTIFIER", type_str)])
            
        # Se não for um tipo conhecido, usar o valor original
        return Tree("SimpleType", [Token("IDENTIFIER", type_str)])
    
    @v_args(inline=True)
    def vector_type(self, element_type, size_expr=None):
        return Tree("VectorType", [element_type, size_expr] if size_expr else [element_type])
    
    @v_args(inline=True)
    def matrix_type(self, element_type, rows_expr, cols_expr):
        return Tree("MatrixType", [element_type, rows_expr, cols_expr])
    
    @v_args(inline=True)
    def tensor_type(self, element_type, dimensions=None):
        return Tree("TensorType", [element_type, dimensions] if dimensions else [element_type])
    
    @v_args(inline=True)
    def distribution_type(self, element_type):
        return Tree("DistributionType", [element_type])
    
    @v_args(inline=True)
    def predefined_distribution_type(self, dist_name):
        return Tree("PredefinedDistributionType", [dist_name])
    
    @v_args(inline=True)
    def space_type(self, element_type):
        return Tree("SpaceType", [element_type])
        
    @v_args(inline=True)
    def predefined_space_type(self, space_name):
        return Tree("PredefinedSpaceType", [space_name])
    
    @v_args(inline=True)
    def custom_type(self, type_name=None):
        if type_name is None:
            logger.warning("custom_type called without type_name, returning Unknown")
            return Tree("CustomType", [Token("IDENTIFIER", "Unknown")])
        return Tree("CustomType", [Token("IDENTIFIER", str(type_name))])
        
    # --- Métodos para Tipos Genéricos ---
    @v_args(inline=True)
    def generic_type(self, type_name, type_params):
        # type_params is already a list from type_param_list
        return Tree("GenericType", [type_name] + (type_params.children if hasattr(type_params, 'children') else [type_params]))
    
    def type_param_list(self, *type_params):
        return Tree("TypeParamList", list(type_params))

    def dimension_list(self, *dims):
        return Tree("DimensionList", list(dims))

    @v_args(inline=True)
    def distribution_type_name(self, name):
        return str(name)

    @v_args(inline=True)
    def space_type_name(self, name):
        return str(name)

    # --- Métodos para Expressões --- 
    @v_args(inline=True)
    def variable_ref(self, identifier1, identifier2=None):
        if identifier2:
            return {"type": "VariableReference", "name": f"{str(identifier1)}.{str(identifier2)}"}
        else:
            return {"type": "VariableReference", "name": str(identifier1)}

    @v_args(inline=True)
    def binary_expression(self, left, op, right):
        return {"type": "BinaryExpression", "operator": str(op), "left": left, "right": right}

    @v_args(inline=True)
    def unary_expression(self, op, operand):
        return {"type": "UnaryExpression", "operator": str(op), "operand": operand}

    # Pass-through for term and factor if they don't add structure
    def term(self, t):
        return t[0] if isinstance(t, list) and len(t) == 1 else t
    
    def factor(self, f):
        return f[0] if isinstance(f, list) and len(f) == 1 else f

    @v_args(inline=True)
    def function_call(self, func_name, args=None):
        return {"type": "FunctionCall", "name": str(func_name), "arguments": args or []}

    def arg_list(self, *args):
        return list(args)

    def list_expr(self, *items):
        return {"type": "ListExpression", "items": list(items)}

    def dict_expr(self, *items):
        return {"type": "DictExpression", "items": list(items)}

    @v_args(inline=True)
    def key_value_pair(self, key, value):
        return {"type": "KeyValuePair", "key": key, "value": value}

    # --- Métodos para Imports e Módulos ---
    @v_args(inline=True)
    def import_statement(self, import_target):
        return {
            "type": "ImportStatement",
            "target": import_target
        }
    
    @v_args(inline=True)
    def import_target(self, module_path, import_list=None):
        if import_list:
            return {
                "type": "ImportTarget",
                "module_path": module_path,
                "import_list": import_list
            }
        else:
            return {
                "type": "ImportTarget",
                "module_path": module_path,
                "import_list": None
            }
    
    @v_args(inline=True)
    def module_path(self, *identifiers):
        return {
            "type": "ModulePath",
            "path": ".".join(str(id) for id in identifiers)
        }
    
    def import_list(self, *identifiers):
        return [str(id) for id in identifiers]
    
    # Removendo o método __default__ para evitar interferências inesperadas
    # e forçar o uso apenas dos métodos explicitamente definidos
