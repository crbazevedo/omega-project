"""
Módulo para o ΩArchitect.

Responsável por expandir a AST do ΩMetaLang para a representação ΩLang.

Autor: Equipe Omega
Data: 28/04/2025
"""

import logging
from typing import Dict, Any, List

# Configuração de logging
logger = logging.getLogger(__name__)

# --- Definição da Estrutura ΩLang (Inicial) ---
# Usaremos dicionários aninhados por enquanto, podemos refatorar para classes depois.
# Exemplo:
# { 
#   "name": "SimpleTest",
#   "variables": {
#       "x": {"name": "x", "type": {"base_type": "Int"}, "initial_value": 10, "scope": "global"},
#       "y": {"name": "y", "type": {"base_type": "Vector", "element_type": {"base_type": "Float"}, "size": 5}, "scope": "global"}
#   },
#   "models": { ... },
#   "environments": { ... },
#   "actions": { ... },
#   "objective": { ... }
# }

class OmegaArchitect:
    """
    Expande a AST do ΩMetaLang para a representação mais detalhada ΩLang.
    
    Processa a árvore sintática abstrata (AST) gerada pelo parser e a transforma
    em uma estrutura de dados que representa o programa de forma mais explícita,
    resolvendo referências e adicionando informações semânticas quando possível.
    """

    def __init__(self):
        """Inicializa o ΩArchitect."""
        logger.info("ΩArchitect inicializado.")
        # No futuro, pode carregar informações da ΩIntelligenceBase

    def expand_ast(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Método principal para expandir a AST ΩMetaLang para ΩLang.

        Args:
            ast: A Árvore Sintática Abstrata (AST) gerada pelo OmegaMetaLangParser.

        Returns:
            Uma representação ΩLang do programa (dicionário Python).
        """
        if not ast or ast.get("type") != "OmegaProgram":
            logger.error("AST inválida ou não é um OmegaProgram.")
            raise ValueError("AST inválida fornecida ao ΩArchitect.")

        omega_lang_repr = {
            "name": ast.get("header", {}).get("name", "UnknownProgram"),
            "variables": {},
            "domains": {},
            "models": {},
            "environments": {},
            "actions": {},
            "objective": None # Será processado
        }

        logger.info(f"Iniciando expansão da AST para o programa: {omega_lang_repr['name']}")

        # Processar declarações
        declarations = ast.get("declarations", {}).get("items", [])
        for decl in declarations:
            decl_type = decl.get("type")
            if decl_type == "VariableDeclaration":
                self._process_variable_declaration(decl, omega_lang_repr)
            elif decl_type == "DomainDeclaration":
                self._process_domain_declaration(decl, omega_lang_repr)
            elif decl_type == "ModelDeclaration":
                self._process_model_declaration(decl, omega_lang_repr)
            elif decl_type == "EnvironmentDeclaration":
                self._process_environment_declaration(decl, omega_lang_repr)
            elif decl_type == "ActionDeclaration":
                self._process_action_declaration(decl, omega_lang_repr)
            else:
                logger.warning(f"Tipo de declaração desconhecido encontrado: {decl_type}")

        # Processar objetivo
        objective_ast = ast.get("objective")
        if objective_ast:
            omega_lang_repr["objective"] = self._process_objective(objective_ast, omega_lang_repr)
        
        logger.info(f"Expansão da AST concluída para: {omega_lang_repr['name']}")
        # TODO: Adicionar validações semânticas na representação ΩLang
        return omega_lang_repr

    # --- Métodos de Processamento Privados ---

    def _process_variable_declaration(self, decl: Dict[str, Any], lang_repr: Dict[str, Any]):
        """Processa uma declaração de variável da AST."""
        var_name = decl.get("name")
        if not var_name:
            logger.warning("Declaração de variável sem nome encontrada.")
            return
        
        # ATENÇÃO: O parser atual tem problemas com a estrutura de 'var_type'.
        # Vamos acessar com cuidado e talvez adicionar fallbacks.
        var_type_info = decl.get("var_type", {"type": "Type", "base_type": "UnknownType", "error": "Type info missing from AST"})
        
        # Simplificação temporária devido ao bug do parser
        if isinstance(var_type_info, str): # Se o transformer retornou só a string (errado)
             logger.warning(f"Tipo da variável '{var_name}' retornado como string: {var_type_info}. Assumindo como base_type.")
             var_type_info = {"type": "Type", "base_type": var_type_info}
        elif not isinstance(var_type_info, dict) or "type" not in var_type_info:
             logger.error(f"Estrutura de tipo inválida para variável '{var_name}': {var_type_info}. Marcando como UnknownType.")
             var_type_info = {"type": "Type", "base_type": "UnknownType", "error": "Invalid type structure from parser"}

        variable_entry = {
            "name": var_name,
            "type": var_type_info, 
            "initial_value": decl.get("initial_value"), # Pode ser None
            "scope": "global" # Assumindo global por enquanto
            # TODO: Adicionar mais informações (ex: descrição, unidade)
        }
        lang_repr["variables"][var_name] = variable_entry
        logger.debug(f"Variável processada: {var_name}")

    def _process_domain_declaration(self, decl: Dict[str, Any], lang_repr: Dict[str, Any]):
        """Processa uma declaração de domínio da AST."""
        domain_name = decl.get("name")
        if not domain_name:
            logger.warning("Declaração de domínio sem nome encontrada.")
            return
        
        items = {}
        for item in decl.get("items", []):
            item_name = item.get("name")
            if item_name:
                # TODO: Processar/validar a expressão do valor
                items[item_name] = {"name": item_name, "value_expr": item.get("value")}
            
        domain_entry = {
            "name": domain_name,
            "items": items
        }
        lang_repr["domains"][domain_name] = domain_entry
        logger.debug(f"Domínio processado: {domain_name}")

    def _process_model_declaration(self, decl: Dict[str, Any], lang_repr: Dict[str, Any]):
        """Processa uma declaração de modelo da AST."""
        model_name = decl.get("name")
        if not model_name:
            logger.warning("Declaração de modelo sem nome encontrada.")
            return

        model_entry = {
            "name": model_name,
            "inputs": [],
            "outputs": [],
            "parameters": {},
            "structure": None,
            "loss": None
            # TODO: Adicionar metadados, referências à IntelligenceBase
        }

        for item in decl.get("items", []):
            item_type = item.get("type")
            if item_type == "ModelInput":
                # TODO: Resolver a referência da variável
                model_entry["inputs"].append(item.get("variable")) 
            elif item_type == "ModelOutput":
                # TODO: Resolver a referência da variável
                model_entry["outputs"].append(item.get("variable"))
            elif item_type == "ModelParameter":
                param_name = item.get("name")
                if param_name:
                     # TODO: Processar/validar o tipo do parâmetro (pode ter o mesmo bug do parser)
                    model_entry["parameters"][param_name] = {"name": param_name, "type": item.get("param_type")}
            elif item_type == "ModelStructure":
                # TODO: Processar a estrutura (predefinida ou customizada)
                model_entry["structure"] = item.get("structure")
            elif item_type == "ModelLoss":
                # TODO: Processar a expressão da loss function
                model_entry["loss"] = item.get("expression")
            else:
                logger.warning(f"Item de modelo desconhecido encontrado em '{model_name}': {item_type}")

        lang_repr["models"][model_name] = model_entry
        logger.debug(f"Modelo processado: {model_name}")

    def _process_environment_declaration(self, decl: Dict[str, Any], lang_repr: Dict[str, Any]):
        """Processa uma declaração de ambiente da AST."""
        env_name = decl.get("name")
        if not env_name:
            logger.warning("Declaração de ambiente sem nome encontrada.")
            return
        
        # TODO: Implementar lógica de processamento para Environment
        env_entry = {"name": env_name, "items": decl.get("items", [])} # Placeholder
        lang_repr["environments"][env_name] = env_entry
        logger.debug(f"Ambiente processado (placeholder): {env_name}")

    def _process_action_declaration(self, decl: Dict[str, Any], lang_repr: Dict[str, Any]):
        """Processa uma declaração de ação da AST."""
        action_name = decl.get("name")
        if not action_name:
            logger.warning("Declaração de ação sem nome encontrada.")
            return
            
        # TODO: Implementar lógica de processamento para Action
        action_entry = {"name": action_name, "items": decl.get("items", [])} # Placeholder
        lang_repr["actions"][action_name] = action_entry
        logger.debug(f"Ação processada (placeholder): {action_name}")

    def _process_objective(self, objective_ast: Dict[str, Any], lang_repr: Dict[str, Any]) -> Dict[str, Any]:
        """Processa a seção de objetivo da AST."""
        objective_entry = {
            "minimize": [],
            "maximize": [],
            "constraints": [],
            "weights": {}
        }
        
        for item in objective_ast.get("items", []):
            item_type = item.get("type")
            expr = item.get("expression") # TODO: Processar/validar a expressão
            if item_type == "ObjectiveMinimize":
                objective_entry["minimize"].append(expr)
            elif item_type == "ObjectiveMaximize":
                objective_entry["maximize"].append(expr)
            elif item_type == "ObjectiveConstraint":
                objective_entry["constraints"].append(expr)
            elif item_type == "ObjectiveWeightS":
                objective_entry["weights"]["S"] = expr # λ
            elif item_type == "ObjectiveWeightA":
                objective_entry["weights"]["A"] = expr # β
            elif item_type == "ObjectiveWeightE":
                objective_entry["weights"]["E"] = expr # μ
            else:
                 logger.warning(f"Item de objetivo desconhecido encontrado: {item_type}")
                 
        logger.debug("Objetivo processado.")
        return objective_entry

# Exemplo de uso (requer uma AST válida do parser)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Carregar AST de exemplo (ou usar o parser)
    # Exemplo simplificado de AST
    example_ast = {
        "type": "OmegaProgram",
        "header": {"type": "Header", "name": "SimpleTest"},
        "declarations": {
            "type": "Declarations",
            "items": [
                {
                    "type": "VariableDeclaration",
                    "name": "x",
                    "var_type": {"type": "Type", "base_type": "Int"}, # Estrutura correta esperada
                    "initial_value": {"type": "Literal", "value": 10, "literal_type": "Int"}
                },
                {
                    "type": "VariableDeclaration",
                    "name": "y",
                    "var_type": {"type": "Type", "base_type": "Vector", "element_type": {"type": "Type", "base_type": "Float"}, "size": {"type": "Literal", "value": 5, "literal_type": "Int"}},
                    "initial_value": None
                },
                 {
                    "type": "ModelDeclaration",
                    "name": "MyModel",
                    "items": [
                        {"type": "ModelInput", "variable": {"type": "VariableRef", "name": "x"}},
                        {"type": "ModelOutput", "variable": {"type": "VariableRef", "name": "y"}},
                        {"type": "ModelStructure", "structure": {"type": "PredefinedModel", "model_type": "LinearRegression"}}
                    ]
                }
            ]
        },
        "objective": {
            "type": "Objective",
            "items": [
                {"type": "ObjectiveMinimize", "expression": {"type": "VariableRef", "name": "S"}},
                {"type": "ObjectiveWeightA", "expression": {"type": "Literal", "value": 1.0, "literal_type": "Float"}}
            ]
        }
    }

    architect = OmegaArchitect()
    try:
        omega_lang_representation = architect.expand_ast(example_ast)
        print("\n--- Representação ΩLang Gerada ---")
        import json
        print(json.dumps(omega_lang_representation, indent=2))
    except ValueError as e:
        print(f"Erro ao expandir AST: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")


