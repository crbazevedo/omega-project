# Proposta de Melhorias para o Parser ΩMetaLang

## Visão Geral

Este documento propõe melhorias específicas para o parser ΩMetaLang, visando superar as limitações identificadas no documento `parser_limitations.md`. As melhorias propostas permitirão que o parser suporte a expressividade completa da linguagem conforme demonstrada nos exemplos avançados do Sprint 4.

## Melhorias Propostas

### 1. Reestruturação Arquitetural

#### 1.1 Modularização da Gramática
- **Proposta**: Dividir a gramática em módulos lógicos (tipos, expressões, declarações, etc.)
- **Implementação**: Criar arquivos separados para cada módulo e um mecanismo de composição
- **Benefício**: Facilita manutenção, teste e extensão de partes específicas da gramática

```python
# Exemplo de estrutura modular
from .grammar_modules.types import types_grammar
from .grammar_modules.expressions import expressions_grammar
from .grammar_modules.declarations import declarations_grammar

omega_grammar = f"""
    // Composição dos módulos
    {types_grammar}
    {expressions_grammar}
    {declarations_grammar}
    
    // Regras de integração
    omega_program: header? import_statement* declaration+
"""
```

#### 1.2 API Pública Estável
- **Proposta**: Definir uma API pública clara para o parser
- **Implementação**: Criar um módulo `parser.py` que exponha apenas interfaces estáveis
- **Benefício**: Facilita integração com outros componentes e testes

```python
# parser.py - API pública
from .internal.grammar import omega_grammar
from .internal.transformer import OmegaMetaLangTransformer

def parse_file(file_path):
    """Parse um arquivo ΩMetaLang e retorna a AST."""
    with open(file_path, 'r') as f:
        return parse_string(f.read())

def parse_string(code):
    """Parse uma string ΩMetaLang e retorna a AST."""
    from lark import Lark
    parser = Lark(omega_grammar, start='omega_program', parser='lalr', 
                 transformer=OmegaMetaLangTransformer())
    return parser.parse(code)
```

### 2. Extensões de Sintaxe

#### 2.1 Suporte a Tipos Complexos
- **Proposta**: Estender a gramática para suportar tipos genéricos e aninhados
- **Implementação**: Modificar as regras de tipos para permitir parâmetros de tipo

```
// Gramática atual (simplificada)
type: IDENTIFIER

// Gramática proposta
type: simple_type | generic_type
simple_type: IDENTIFIER
generic_type: IDENTIFIER "<" type ("," type)* ">"
```

#### 2.2 Suporte a Modelos Causais
- **Proposta**: Adicionar regras para declarações de modelos causais
- **Implementação**: Criar novas regras para `causal_model` e estruturas relacionadas

```
// Novas regras para modelos causais
declaration: ... | causal_model_decl
causal_model_decl: "causal_model" IDENTIFIER "{" causal_model_body "}"
causal_model_body: variables_block? structure_block? mechanisms_block? function_decl*
variables_block: "variables" ":" "{" variable_decl+ "}" ";"
structure_block: "structure" ":" "{" edge_decl+ "}" ";"
edge_decl: IDENTIFIER "->" IDENTIFIER ","?
```

#### 2.3 Suporte a Garantias Formais
- **Proposta**: Adicionar regras para especificação de garantias formais
- **Implementação**: Criar novas regras para blocos `guarantees` e propriedades

```
// Novas regras para garantias formais
class_body: ... | guarantees_block
guarantees_block: "guarantees" ":" "{" guarantee_decl+ "}"
guarantee_decl: IDENTIFIER ":" "{" guarantee_property+ "}" ","?
guarantee_property: IDENTIFIER ":" value ","?
```

#### 2.4 Operadores Especiais
- **Proposta**: Adicionar suporte para operadores de inferência causal
- **Implementação**: Estender as regras de expressão para incluir operadores como `do()`

```
// Novas regras para operadores causais
atom: ... | do_operator
do_operator: "do" "(" IDENTIFIER "," expression ")"
```

### 3. Melhorias Semânticas

#### 3.1 Sistema de Tipos Aprimorado
- **Proposta**: Implementar um sistema de tipos mais robusto
- **Implementação**: Criar classes para representar tipos e um verificador de tipos
- **Benefício**: Detecção precoce de erros de tipo e melhor suporte a IDEs

```python
class Type:
    """Classe base para todos os tipos."""
    pass

class SimpleType(Type):
    """Tipo simples como Int, Float, etc."""
    def __init__(self, name):
        self.name = name

class GenericType(Type):
    """Tipo genérico como Vector<Float>, Distribution<Int>, etc."""
    def __init__(self, base_type, type_params):
        self.base_type = base_type
        self.type_params = type_params
```

#### 3.2 Resolução de Escopo Melhorada
- **Proposta**: Implementar um resolvedor de escopo mais sofisticado
- **Implementação**: Criar uma estrutura de tabela de símbolos hierárquica
- **Benefício**: Melhor suporte para referências cruzadas e verificação de nomes

```python
class Scope:
    """Representa um escopo com símbolos definidos."""
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent
    
    def define(self, name, symbol):
        """Define um símbolo neste escopo."""
        self.symbols[name] = symbol
    
    def resolve(self, name):
        """Resolve um nome neste escopo ou em escopos pais."""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.resolve(name)
        return None
```

#### 3.3 Transformação de AST Robusta
- **Proposta**: Refatorar o transformer para ser mais robusto e extensível
- **Implementação**: Usar padrão visitor e tratamento de erros melhorado
- **Benefício**: Transformação mais confiável e mensagens de erro mais claras

```python
class ASTVisitor:
    """Visitor base para percorrer a AST."""
    def visit(self, node):
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name, self.generic_visit)
        return method(node)
    
    def generic_visit(self, node):
        """Método padrão para nós sem visitante específico."""
        if hasattr(node, "children"):
            for child in node.children:
                self.visit(child)
```

### 4. Melhorias de Robustez

#### 4.1 Tratamento de Erros Aprimorado
- **Proposta**: Implementar tratamento de erros mais informativo
- **Implementação**: Criar classes de erro específicas e mensagens contextuais
- **Benefício**: Depuração mais fácil e melhor experiência do desenvolvedor

```python
class OmegaParseError(Exception):
    """Erro base para problemas de parsing."""
    def __init__(self, message, line, column, context=None):
        self.message = message
        self.line = line
        self.column = column
        self.context = context
        super().__init__(self.format_message())
    
    def format_message(self):
        """Formata a mensagem de erro com contexto."""
        msg = f"Erro na linha {self.line}, coluna {self.column}: {self.message}"
        if self.context:
            msg += f"\nContexto: {self.context}"
        return msg
```

#### 4.2 Testes Abrangentes
- **Proposta**: Criar uma suíte de testes abrangente
- **Implementação**: Testes unitários para cada componente e testes de integração
- **Benefício**: Maior confiabilidade e facilidade de manutenção

```python
def test_generic_types():
    """Testa parsing de tipos genéricos."""
    code = "model MyModel<Vector<Float>, Distribution<Int>> { ... }"
    ast = parse_string(code)
    assert ast.type_params[0].name == "Vector"
    assert ast.type_params[0].params[0].name == "Float"
    assert ast.type_params[1].name == "Distribution"
    assert ast.type_params[1].params[0].name == "Int"
```

#### 4.3 Modo de Recuperação de Erros
- **Proposta**: Implementar modo de recuperação para continuar parsing após erros
- **Implementação**: Usar recursos de recuperação do Lark e estratégias personalizadas
- **Benefício**: Melhor feedback para múltiplos erros em um único arquivo

```python
# Configuração do parser com recuperação de erros
parser = Lark(omega_grammar, start='omega_program', parser='lalr',
             transformer=OmegaMetaLangTransformer(),
             propagate_positions=True,
             maybe_placeholders=True)
```

## Plano de Implementação

### Fase 1: Reestruturação Arquitetural
1. Refatorar a estrutura do código para modularizar a gramática
2. Definir e implementar a API pública estável
3. Criar testes para a nova estrutura

### Fase 2: Extensões de Sintaxe
1. Implementar suporte a tipos complexos
2. Adicionar suporte a modelos causais
3. Implementar suporte a garantias formais
4. Adicionar operadores especiais

### Fase 3: Melhorias Semânticas
1. Implementar sistema de tipos aprimorado
2. Desenvolver resolução de escopo melhorada
3. Refatorar transformação de AST

### Fase 4: Melhorias de Robustez
1. Implementar tratamento de erros aprimorado
2. Criar suíte de testes abrangente
3. Adicionar modo de recuperação de erros

## Conclusão

As melhorias propostas transformarão o parser ΩMetaLang em um componente robusto e extensível, capaz de processar a expressividade completa da linguagem conforme demonstrada nos exemplos avançados. Estas melhorias não apenas resolverão as limitações atuais, mas também estabelecerão uma base sólida para futuras extensões da linguagem.

A implementação destas melhorias deve ser priorizada para o Sprint 5, com foco inicial na reestruturação arquitetural e nas extensões de sintaxe necessárias para suportar os exemplos do Sprint 4.
