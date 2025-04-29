"""
API pública do parser ΩMetaLang.

Este módulo fornece uma API pública estável para o parser ΩMetaLang,
abstraindo os detalhes de implementação e fornecendo uma interface
consistente para os usuários.

Autor: Equipe Omega
Data: 28/04/2025
"""

from lark import Lark, ParseError, Tree, Token
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import json
import os
import sys

from .grammar_modules.grammar import OMEGA_METALANG_GRAMMAR
from .transformer import OmegaMetaLangTransformer

# Configuração de logging
logger = logging.getLogger(__name__)

class ParserError(Exception):
    """Exceção personalizada para erros de parsing."""
    
    def __init__(self, message, line=None, column=None, context=None):
        self.message = message
        self.line = line
        self.column = column
        self.context = context
        super().__init__(self.format_message())
    
    def format_message(self):
        """Formata a mensagem de erro com informações de localização."""
        if self.line is not None and self.column is not None:
            return f"Erro de parsing na linha {self.line}, coluna {self.column}: {self.message}"
        return f"Erro de parsing: {self.message}"

class ParseResult:
    """
    Classe que encapsula o resultado da análise sintática.
    
    Fornece métodos para acessar e manipular a AST resultante.
    """
    
    def __init__(self, ast: Dict[str, Any], source_path: Optional[str] = None):
        self.ast = ast
        self.source_path = source_path
        
    def _tree_to_dict(self, node: Union[Tree, Token, Any]) -> Any:
        """Converte recursivamente um nó Tree/Token ou valor em um dicionário/valor."""
        if node is None:
            return None
            
        if isinstance(node, Tree):
            # Tratar nós Tree
            if node.data == "SimpleType" and node.children:
                return {"type": "SimpleType", "name": node.children[0].value if hasattr(node.children[0], 'value') else str(node.children[0])}
            elif node.data == "VectorType":
                element_type = self._tree_to_dict(node.children[0]) if node.children else None
                size = self._tree_to_dict(node.children[1]) if len(node.children) > 1 else None
                return {"type": "VectorType", "element_type": element_type, "size": size}
            elif node.data == "MatrixType" and len(node.children) >= 3:
                return {"type": "MatrixType", 
                        "element_type": self._tree_to_dict(node.children[0]), 
                        "rows": self._tree_to_dict(node.children[1]), 
                        "cols": self._tree_to_dict(node.children[2])}
            elif node.data == "TensorType":
                element_type = self._tree_to_dict(node.children[0]) if node.children else None
                dimensions = [self._tree_to_dict(dim) for dim in node.children[1:]] if len(node.children) > 1 else []
                return {"type": "TensorType", "element_type": element_type, "dimensions": dimensions}
            elif node.data == "DistributionType" and node.children:
                return {"type": "DistributionType", "element_type": self._tree_to_dict(node.children[0])}
            elif node.data == "PredefinedDistributionType" and node.children:
                return {"type": "PredefinedDistributionType", "name": node.children[0].value if hasattr(node.children[0], 'value') else str(node.children[0])}
            elif node.data == "SpaceType" and node.children:
                return {"type": "SpaceType", "element_type": self._tree_to_dict(node.children[0])}
            elif node.data == "PredefinedSpaceType" and node.children:
                return {"type": "PredefinedSpaceType", "name": node.children[0].value if hasattr(node.children[0], 'value') else str(node.children[0])}
            elif node.data == "CustomType" and node.children:
                return {"type": "CustomType", "name": node.children[0].value if hasattr(node.children[0], 'value') else str(node.children[0])}
            elif node.data == "GenericType" and len(node.children) > 0:
                type_name = node.children[0].value if hasattr(node.children[0], 'value') else str(node.children[0])
                params = [self._tree_to_dict(p) for p in node.children[1:]]
                return {"type": "GenericType", "name": type_name, "type_params": params}
            elif node.data == "BinaryExpression" and len(node.children) >= 3:
                return {"type": "BinaryExpression", 
                        "operator": node.children[1].value if hasattr(node.children[1], 'value') else str(node.children[1]),
                        "left": self._tree_to_dict(node.children[0]),
                        "right": self._tree_to_dict(node.children[2])}
            elif node.data == "FunctionCall":
                func_name = node.children[0].value if hasattr(node.children[0], 'value') else str(node.children[0])
                args = [self._tree_to_dict(arg) for arg in node.children[1:]] if len(node.children) > 1 else []
                return {"type": "FunctionCall", "name": func_name, "arguments": args}
            elif node.data == "Literal":
                value = node.children[0].value if hasattr(node.children[0], 'value') else str(node.children[0])
                try:
                    # Tentar converter para número se possível
                    if '.' in value:
                        return {"type": "Literal", "value": float(value)}
                    else:
                        return {"type": "Literal", "value": int(value)}
                except (ValueError, TypeError):
                    # Se não for um número, manter como string
                    return {"type": "Literal", "value": value}
            else:
                # Estrutura genérica para outros tipos de nós
                return {"type": node.data, "children": [self._tree_to_dict(child) for child in node.children]}
                
        elif isinstance(node, Token):
            # Tratar nós Token
            if node.type == "NUMBER":
                try:
                    return {"value": float(node.value) if '.' in node.value else int(node.value)}
                except ValueError:
                    return {"value": node.value}
            elif node.type == "STRING":
                return {"value": node.value.strip('"')}
            else:
                return node.value
        elif isinstance(node, list):
            # Tratar listas (como argumentos de função)
            return [self._tree_to_dict(item) for item in node]
        elif isinstance(node, dict):
            # Já é um dicionário, retornar como está
            return node
        else:
            # Outros tipos primitivos
            return node
    
    def get_program_name(self) -> str:
        """Retorna o nome do programa ΩMetaLang."""
        try:
            return self.ast["header"]["name"]
        except (KeyError, TypeError):
            return "UnknownProgram"
    
    def get_declarations(self) -> List[Dict[str, Any]]:
        """Retorna a lista de declarações no programa."""
        try:
            # Verificar a estrutura da AST
            if isinstance(self.ast, dict) and "children" in self.ast:
                # Estrutura nova: AST é um dicionário com 'children'
                header_node = None
                declarations_node = None
                
                # Procurar o nó de declarações
                for child in self.ast["children"]:
                    if isinstance(child, dict):
                        if child.get("type") == "Header":
                            header_node = child
                        elif child.get("type") == "Declarations":
                            declarations_node = child
                
                if declarations_node and "items" in declarations_node:
                    # Extrair as declarações do nó encontrado
                    raw_declarations = declarations_node["items"]
                    declarations = []
                    
                    # Processar cada declaração, que pode estar aninhada
                    for decl in raw_declarations:
                        if isinstance(decl, dict) and "children" in decl:
                            # Declaração aninhada (ex: declaration > variable_decl)
                            for child in decl["children"]:
                                if isinstance(child, dict) and "children" in child:
                                    # Extrair a declaração de variável
                                    if isinstance(child["type"], Token) and child["type"].value == "variable_decl":
                                        var_name = child["children"][0]
                                        var_type = child["children"][1]
                                        var_dict = {
                                            "type": "VariableDeclaration",
                                            "name": var_name,
                                            "var_type": var_type
                                        }
                                        # Adicionar valor inicial se existir
                                        if len(child["children"]) > 2:
                                            var_dict["initial_value"] = child["children"][2]
                                        declarations.append(var_dict)
                    
                    return declarations
            
            # Fallback para a estrutura antiga
            declarations = self.ast.get("declarations", {}).get("items", [])
            # Verificar se as declarações são objetos Tree e convertê-los se necessário
            if declarations and any(isinstance(decl, Tree) for decl in declarations):
                return [self._tree_to_dict(decl) if isinstance(decl, Tree) else decl for decl in declarations]
            return declarations
        except (KeyError, TypeError) as e:
            logger.warning(f"Erro ao extrair declarações: {e}")
            return []
    
    def get_objective(self) -> Dict[str, Any]:
        """Retorna o objetivo do programa."""
        try:
            objective = self.ast["objective"]
            # Verificar se o objetivo é um objeto Tree e convertê-lo se necessário
            if isinstance(objective, Tree):
                return self._tree_to_dict(objective)
            return objective
        except (KeyError, TypeError):
            return {"type": "Objective", "items": []}
    
    def get_variables(self) -> List[Dict[str, Any]]:
        """Retorna a lista de declarações de variáveis."""
        declarations = self.get_declarations()
        variables = []
        for decl in declarations:
            if isinstance(decl, dict) and decl.get("type") == "VariableDeclaration":
                variables.append(decl)
            elif hasattr(decl, 'data') and decl.data == "VariableDeclaration":
                # Converter Tree para dicionário para compatibilidade com testes
                var_dict = {
                    "type": "VariableDeclaration",
                    "name": decl.children[0].value if hasattr(decl.children[0], 'value') else str(decl.children[0]),
                    "var_type": self._tree_to_dict(decl.children[1]),
                }
                # Verificar se há valor inicial
                if len(decl.children) > 2:
                    var_dict["initial_value"] = self._tree_to_dict(decl.children[2])
                variables.append(var_dict)
        return variables
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Retorna a lista de declarações de modelos."""
        return [decl for decl in self.get_declarations() 
                if decl.get("type") == "ModelDeclaration"]
    
    def get_environments(self) -> List[Dict[str, Any]]:
        """Retorna a lista de declarações de ambientes."""
        return [decl for decl in self.get_declarations() 
                if decl.get("type") == "EnvironmentDeclaration"]
    
    def get_actions(self) -> List[Dict[str, Any]]:
        """Retorna a lista de declarações de ações."""
        return [decl for decl in self.get_declarations() 
                if decl.get("type") == "ActionDeclaration"]
    
    def get_domains(self) -> List[Dict[str, Any]]:
        """Retorna a lista de declarações de domínios."""
        return [decl for decl in self.get_declarations() 
                if decl.get("type") == "DomainDeclaration"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Retorna a AST como um dicionário Python."""
        return self.ast
    
    def to_json(self, indent: int = 2) -> str:
        """Retorna a AST como uma string JSON formatada."""
        return json.dumps(self.ast, indent=indent)
    
    def save_json(self, output_path: str, indent: int = 2) -> None:
        """Salva a AST em um arquivo JSON."""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.ast, f, indent=indent)
            logger.info(f"AST salva em: {output_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar AST: {e}")
            raise

class OmegaMetaLangParser:
    """
    Parser para a linguagem ΩMetaLang.
    
    Esta classe fornece métodos para analisar código ΩMetaLang e
    gerar uma AST (Árvore Sintática Abstrata) estruturada.
    """
    
    def __init__(self, debug: bool = False):
        """
        Inicializa o parser com a gramática definida.
        
        Args:
            debug: Se True, habilita o modo de depuração com mensagens detalhadas
        """
        self.debug = debug
        try:
            # Configurar o nível de logging com base no modo de depuração
            log_level = logging.DEBUG if debug else logging.INFO
            logging.basicConfig(level=log_level)
            
            # Usar LALR para eficiência, mas pode exigir ajustes na gramática se houver conflitos
            self.parser = Lark(
                OMEGA_METALANG_GRAMMAR, 
                parser='lalr', 
                transformer=OmegaMetaLangTransformer()
            )
            logger.info("Parser ΩMetaLang inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar o parser ΩMetaLang: {e}")
            raise ParserError(f"Falha na inicialização do parser: {e}")
    
    def parse(self, text: str) -> ParseResult:
        """
        Analisa um texto em ΩMetaLang e retorna um objeto ParseResult.
        
        Args:
            text: Código fonte em ΩMetaLang
            
        Returns:
            ParseResult contendo a AST e métodos para manipulá-la
            
        Raises:
            ParserError: Se ocorrer um erro durante a análise sintática
        """
        try:
            # Obter a AST bruta do parser
            raw_ast = self.parser.parse(text)
            
            # Converter a AST para dicionário se necessário
            ast = self._convert_ast_to_dict(raw_ast)
            
            logger.info("Análise sintática concluída com sucesso")
            return ParseResult(ast)
        except ParseError as e:
            # Extrair informações detalhadas do erro
            line, column = e.line, e.column
            context = text.splitlines()[line-1] if line <= len(text.splitlines()) else ""
            logger.error(f"Erro na análise sintática: linha {line}, coluna {column}")
            raise ParserError(str(e), line, column, context)
        except Exception as e:
            logger.error(f"Erro inesperado na análise sintática: {e}")
            raise ParserError(f"Erro inesperado: {e}")
    
    def _convert_ast_to_dict(self, node: Union[Tree, Token, Dict, List, Any]) -> Any:
        """
        Converte recursivamente uma AST contendo objetos Tree/Token em dicionários/listas.
        
        Args:
            node: Nó da AST a ser convertido
            
        Returns:
            Versão convertida do nó (dicionário, lista ou valor primitivo)
        """
        if isinstance(node, Tree):
            # Converter Tree para dicionário
            if node.data == "program" and len(node.children) >= 1:
                result = {
                    "header": self._convert_ast_to_dict(node.children[0]),
                }
                
                # Adicionar declarações se existirem
                if len(node.children) > 1:
                    result["declarations"] = self._convert_ast_to_dict(node.children[1])
                else:
                    result["declarations"] = {"type": "Declarations", "items": []}
                
                # Adicionar objetivo se existir
                if len(node.children) > 2:
                    result["objective"] = self._convert_ast_to_dict(node.children[2])
                else:
                    result["objective"] = {"type": "Objective", "items": []}
                
                return result
            elif node.data == "header" and node.children:
                return {"type": "Header", "name": str(node.children[0].value) if hasattr(node.children[0], 'value') else str(node.children[0])}
            elif node.data == "declarations":
                return {"type": "Declarations", "items": [self._convert_ast_to_dict(child) for child in node.children]}
            elif node.data == "objective":
                return {"type": "Objective", "items": [self._convert_ast_to_dict(child) for child in node.children]}
            elif node.data == "VariableDeclaration":
                var_dict = {
                    "type": "VariableDeclaration",
                    "name": node.children[0].value if hasattr(node.children[0], 'value') else str(node.children[0]),
                    "var_type": self._convert_ast_to_dict(node.children[1]),
                }
                # Adicionar valor inicial se existir
                if len(node.children) > 2:
                    var_dict["initial_value"] = self._convert_ast_to_dict(node.children[2])
                return var_dict
            elif node.data == "SimpleType" and node.children:
                return {"type": "SimpleType", "name": node.children[0].value if hasattr(node.children[0], 'value') else str(node.children[0])}
            elif node.data == "VectorType":
                element_type = self._convert_ast_to_dict(node.children[0]) if node.children else None
                size = self._convert_ast_to_dict(node.children[1]) if len(node.children) > 1 else None
                return {"type": "VectorType", "element_type": element_type, "size": size}
            elif node.data == "MatrixType" and len(node.children) >= 3:
                return {"type": "MatrixType", 
                        "element_type": self._convert_ast_to_dict(node.children[0]), 
                        "rows": self._convert_ast_to_dict(node.children[1]), 
                        "cols": self._convert_ast_to_dict(node.children[2])}
            elif node.data == "TensorType":
                element_type = self._convert_ast_to_dict(node.children[0]) if node.children else None
                dimensions = [self._convert_ast_to_dict(dim) for dim in node.children[1:]] if len(node.children) > 1 else []
                return {"type": "TensorType", "element_type": element_type, "dimensions": dimensions}
            elif node.data == "DistributionType" and node.children:
                return {"type": "DistributionType", "element_type": self._convert_ast_to_dict(node.children[0])}
            elif node.data == "PredefinedDistributionType" and node.children:
                return {"type": "PredefinedDistributionType", "name": node.children[0].value if hasattr(node.children[0], 'value') else str(node.children[0])}
            elif node.data == "SpaceType" and node.children:
                return {"type": "SpaceType", "element_type": self._convert_ast_to_dict(node.children[0])}
            elif node.data == "PredefinedSpaceType" and node.children:
                return {"type": "PredefinedSpaceType", "name": node.children[0].value if hasattr(node.children[0], 'value') else str(node.children[0])}
            elif node.data == "CustomType" and node.children:
                return {"type": "CustomType", "name": node.children[0].value if hasattr(node.children[0], 'value') else str(node.children[0])}
            elif node.data == "GenericType" and len(node.children) > 0:
                type_name = node.children[0].value if hasattr(node.children[0], 'value') else str(node.children[0])
                params = [self._convert_ast_to_dict(p) for p in node.children[1:]]
                return {"type": "GenericType", "name": type_name, "type_params": params}
            elif node.data == "TypeParamList":
                return [self._convert_ast_to_dict(param) for param in node.children]
            elif node.data == "DimensionList":
                return [self._convert_ast_to_dict(dim) for dim in node.children]
            else:
                # Estrutura genérica para outros tipos de nós
                return {"type": node.data, "children": [self._convert_ast_to_dict(child) for child in node.children]}
        elif isinstance(node, Token):
            # Converter Token para valor
            if node.type == "NUMBER":
                try:
                    return float(node.value) if '.' in node.value else int(node.value)
                except ValueError:
                    return node.value
            elif node.type == "STRING":
                return node.value.strip('"')
            else:
                return node.value
        elif isinstance(node, dict):
            # Já é um dicionário, converter valores recursivamente
            return {k: self._convert_ast_to_dict(v) for k, v in node.items()}
        elif isinstance(node, list):
            # Converter cada item da lista recursivamente
            return [self._convert_ast_to_dict(item) for item in node]
        else:
            # Outros tipos primitivos
            return node
    
    def parse_file(self, file_path: str) -> ParseResult:
        """
        Analisa um arquivo em ΩMetaLang e retorna um objeto ParseResult.
        
        Args:
            file_path: Caminho para o arquivo
            
        Returns:
            ParseResult contendo a AST e métodos para manipulá-la
            
        Raises:
            ParserError: Se ocorrer um erro durante a análise sintática
            FileNotFoundError: Se o arquivo não for encontrado
        """
        try:
            with open(file_path, 'r') as f:
                text = f.read()
            
            logger.info(f"Arquivo carregado: {file_path}")
            result = self.parse(text)
            # Adicionar o caminho do arquivo ao resultado
            result.source_path = file_path
            return result
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {file_path}")
            raise
        except ParserError:
            # Repassar erros de parsing sem modificação
            raise
        except Exception as e:
            logger.error(f"Erro ao analisar arquivo {file_path}: {e}")
            raise ParserError(f"Erro ao processar arquivo: {e}")
    
    def validate(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Valida um texto em ΩMetaLang sem gerar a AST completa.
        
        Args:
            text: Código fonte em ΩMetaLang
            
        Returns:
            Tupla (válido, mensagem_erro), onde válido é um booleano indicando
            se o código é válido, e mensagem_erro é None se for válido ou uma
            string descrevendo o erro caso contrário.
        """
        try:
            self.parse(text)
            return True, None
        except ParserError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Erro inesperado: {e}"
    
    def validate_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Valida um arquivo em ΩMetaLang sem gerar a AST completa.
        
        Args:
            file_path: Caminho para o arquivo
            
        Returns:
            Tupla (válido, mensagem_erro), onde válido é um booleano indicando
            se o código é válido, e mensagem_erro é None se for válido ou uma
            string descrevendo o erro caso contrário.
        """
        try:
            self.parse_file(file_path)
            return True, None
        except FileNotFoundError:
            return False, f"Arquivo não encontrado: {file_path}"
        except ParserError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Erro inesperado: {e}"
    
    @staticmethod
    def get_version() -> str:
        """Retorna a versão atual do parser ΩMetaLang."""
        from .grammar_modules import GRAMMAR_VERSION
        return GRAMMAR_VERSION
    
    @staticmethod
    def get_grammar() -> str:
        """Retorna a gramática completa do ΩMetaLang como string."""
        return OMEGA_METALANG_GRAMMAR

# Função de conveniência para criar um parser com configuração padrão
def create_parser(debug: bool = False) -> OmegaMetaLangParser:
    """
    Cria e retorna uma instância do parser ΩMetaLang com configuração padrão.
    
    Args:
        debug: Se True, habilita o modo de depuração
        
    Returns:
        Instância configurada do OmegaMetaLangParser
    """
    return OmegaMetaLangParser(debug=debug)



