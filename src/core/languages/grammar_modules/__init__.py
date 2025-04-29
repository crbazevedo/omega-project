"""
Módulo base para os componentes da gramática ΩMetaLang.

Este módulo contém constantes e configurações compartilhadas por todos os módulos da gramática.

Autor: Equipe Omega
Data: 28/04/2025
"""

import logging

# Configuração de logging
logger = logging.getLogger(__name__)

# Configurações compartilhadas para todos os módulos da gramática
GRAMMAR_VERSION = "1.0.0"
DEBUG_MODE = False

# Configurações de formatação para a gramática
GRAMMAR_HEADER = r"""
    // ΩMetaLang Grammar v{version}
    // Gerado automaticamente a partir de módulos
    
    ?start: omega_program
    
"""

# Configurações de importação comuns para todos os módulos
COMMON_IMPORTS = r"""
    %import common.WS
    %import common.NEWLINE
    %ignore WS
    %ignore NEWLINE
    %ignore /\/\/[^\n]*/  // Comentários de linha
    %ignore /\/\*(.|\n)*?\*\//  // Comentários de bloco
"""

def format_grammar_module(module_name, module_content):
    """
    Formata um módulo de gramática com cabeçalho e comentários.
    
    Args:
        module_name: Nome do módulo
        module_content: Conteúdo da gramática
        
    Returns:
        String formatada do módulo
    """
    return f"""
    // Módulo: {module_name}
    // Versão: {GRAMMAR_VERSION}
    
{module_content}
"""

def log_grammar_loading(module_name):
    """
    Registra o carregamento de um módulo de gramática.
    
    Args:
        module_name: Nome do módulo carregado
    """
    if DEBUG_MODE:
        logger.debug(f"Carregando módulo de gramática: {module_name}")
