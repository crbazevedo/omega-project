# Abordagens Técnicas para o Sprint 3

Este documento detalha as abordagens técnicas recomendadas para as tarefas priorizadas no Sprint 3, fornecendo diretrizes específicas para implementação.

## 1. [P0] US-DT-1.1: Finalizar Correção do Parser ΩMetaLang

### Análise do Problema

O erro atual `TypeError: OmegaMetaLangTransformer.simple_type() missing 1 required positional argument: 'type_name'` sugere um problema na forma como o Lark está passando argumentos para o método `simple_type()` do transformer. Isso pode ocorrer devido a:

1. Incompatibilidade entre a gramática e o transformer
2. Uso incorreto do decorador `@v_args(inline=True)`
3. Mudanças na estrutura da árvore de parse que o transformer não está preparado para processar

### Abordagem Técnica Recomendada

**Estratégia 1: Correção Direta do Transformer**

```python
# Abordagem atual (problemática)
@v_args(inline=True)
def simple_type(self, type_name):
    return {"type": "Type", "base_type": str(type_name)}

# Possível solução 1: Remover o decorador e processar a árvore diretamente
def simple_type(self, tree):
    if len(tree.children) > 0:
        type_name = tree.children[0]
        return {"type": "Type", "base_type": str(type_name)}
    return {"type": "Type", "base_type": "UnknownType"}

# Possível solução 2: Manter o decorador mas tornar o argumento opcional
@v_args(inline=True)
def simple_type(self, type_name=None):
    if type_name is None:
        return {"type": "Type", "base_type": "UnknownType"}
    return {"type": "Type", "base_type": str(type_name)}
```

**Estratégia 2: Refatoração Completa do Sistema de Tipos**

Se a correção direta não funcionar, uma refatoração mais ampla pode ser necessária:

1. Simplificar a gramática para tipos, tornando-a mais direta
2. Implementar um novo transformer específico para tipos
3. Usar um padrão de visitante mais explícito em vez de depender do comportamento automático do Lark

```python
# Exemplo de refatoração da gramática
# Antes
simple_type: IDENTIFIER
# Depois
simple_type: "type" ":" IDENTIFIER

# Transformer correspondente
def simple_type(self, tree):
    # Acesso mais explícito aos nós da árvore
    identifier_token = tree.children[2]  # Pula "type" e ":"
    return {"type": "Type", "base_type": str(identifier_token)}
```

**Estratégia 3: Substituição do Lark**

Como último recurso, se as limitações do Lark se mostrarem muito restritivas:

1. Avaliar parsers alternativos como ANTLR4, PLY ou PyParsing
2. Implementar um parser customizado para a linguagem ΩMetaLang
3. Manter compatibilidade com a AST atual para minimizar impacto em outros componentes

### Passos de Implementação

1. Criar uma branch de desenvolvimento isolada: `git checkout -b sprint3/parser-final-fix`
2. Implementar a Estratégia 1 e testar
3. Se falhar, tentar a Estratégia 2
4. Documentar todas as alterações e decisões tomadas
5. Garantir que todos os testes passem antes de fazer merge

## 2. [P1] US-2.2: Implementar ΩSearch para Recuperação de Artefatos

### Arquitetura Proposta

O ΩSearch deve ser implementado como um componente modular que:

1. Carrega artefatos da ΩIntelligenceBase (mock inicialmente)
2. Aceita consultas na forma de embeddings vetoriais
3. Calcula similaridade entre a consulta e os artefatos
4. Retorna os artefatos mais similares com seus scores

```
┌─────────────┐     ┌───────────────┐     ┌──────────────────┐
│  Embedding  │────▶│   ΩSearch     │────▶│  Top-k Artefatos │
│  de Consulta│     │ (Similaridade)│     │  com Scores      │
└─────────────┘     └───────┬───────┘     └──────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ΩIntelligence- │
                    │    Base       │
                    │  (Mock)       │
                    └───────────────┘
```

### Abordagem Técnica Recomendada

**Estrutura da ΩIntelligenceBase Mock**

Criar uma estrutura de dados JSON para representar artefatos:

```json
[
  {
    "id": "artifact_001",
    "type": "model_template",
    "name": "LogisticRegressionTemplate",
    "description": "Template para regressão logística binária",
    "tags": ["classification", "binary", "supervised"],
    "embedding": [0.1, 0.2, 0.3, ..., 0.768],  // Vetor de 768 dimensões (exemplo)
    "content": {
      "structure": "LogisticRegression",
      "hyperparameters": {
        "learning_rate": 0.01,
        "regularization": "l2"
      }
    }
  },
  // Mais artefatos...
]
```

**Implementação da Busca por Similaridade**

Usar NumPy para cálculos eficientes de similaridade de cosseno:

```python
import numpy as np
from typing import List, Dict, Any

class OmegaSearch:
    def __init__(self, intelligence_base_path: str):
        """Inicializa o ΩSearch carregando artefatos da ΩIntelligenceBase."""
        self.artifacts = self._load_artifacts(intelligence_base_path)
        # Pré-computar matriz de embeddings para busca eficiente
        self.embedding_matrix = np.array([artifact["embedding"] for artifact in self.artifacts])
        
    def _load_artifacts(self, path: str) -> List[Dict[str, Any]]:
        """Carrega artefatos do caminho especificado."""
        import json
        with open(path, 'r') as f:
            return json.load(f)
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Busca os k artefatos mais similares ao embedding de consulta.
        
        Args:
            query_embedding: Vetor de embedding da consulta
            k: Número de resultados a retornar
            
        Returns:
            Lista dos k artefatos mais similares com seus scores
        """
        # Converter para array numpy
        query_vector = np.array(query_embedding)
        
        # Normalizar o vetor de consulta
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        
        # Calcular similaridade de cosseno com todos os artefatos
        # (assumindo que self.embedding_matrix já está normalizada)
        similarities = np.dot(self.embedding_matrix, query_vector)
        
        # Obter os índices dos k artefatos mais similares
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Construir resultados
        results = []
        for idx in top_indices:
            artifact = self.artifacts[idx].copy()
            artifact["similarity_score"] = float(similarities[idx])
            results.append(artifact)
            
        return results
```

**Extensões Futuras**

1. Integração com bibliotecas de busca vetorial como FAISS ou Annoy para escalabilidade
2. Suporte para consultas híbridas (texto + embedding)
3. Indexação incremental de novos artefatos

## 3. [P1] DT-4.2: Documentar Guia de Referência ΩMetaLang

### Estrutura do Documento

O guia de referência deve seguir esta estrutura:

1. **Introdução**
   - Visão geral da linguagem ΩMetaLang
   - Propósito e casos de uso
   - Relação com a equação Omega (S + λC - βA + μE)

2. **Sintaxe Básica**
   - Estrutura de um programa ΩMetaLang
   - Comentários e espaços em branco
   - Identificadores e palavras-chave

3. **Tipos de Dados**
   - Tipos primitivos (Int, Float, String, Bool)
   - Tipos compostos (Vector, Matrix, Tensor)
   - Tipos especiais (Distribution, Space)
   - Tipos personalizados

4. **Declarações**
   - Variáveis
   - Modelos
   - Domínios
   - Ambientes
   - Ações

5. **Expressões**
   - Literais
   - Referências a variáveis
   - Operadores (aritméticos, lógicos, comparação)
   - Chamadas de função

6. **Objetivo**
   - Minimize/Maximize
   - Constraints
   - Weights (S, A, E)

7. **Exemplos Completos**
   - Classificação simples
   - Regressão linear
   - Séries temporais
   - Aprendizado por reforço

### Abordagem Técnica

1. Usar Markdown para formatação rica e fácil manutenção
2. Incluir blocos de código com syntax highlighting
3. Adicionar diagramas de sintaxe para construções complexas (usando notação EBNF ou diagramas de railroad)
4. Criar tabelas para resumir informações (ex: operadores e precedência)

**Exemplo de Seção de Tipos**

```markdown
## Tipos de Dados

### Tipos Primitivos

| Tipo    | Descrição                           | Exemplo Literal |
|---------|-------------------------------------|----------------|
| Int     | Número inteiro                      | 42             |
| Float   | Número de ponto flutuante           | 3.14           |
| String  | Sequência de caracteres             | "texto"        |
| Bool    | Valor booleano (verdadeiro/falso)   | true, false    |

### Tipos Compostos

ΩMetaLang suporta tipos compostos para representar dados estruturados:

#### Vector

Representa uma sequência unidimensional de elementos do mesmo tipo.

**Sintaxe:** `Vector<tipo_elemento, tamanho>`

**Exemplo:**
```omega
Variable v : Vector<Float, 10>;  // Vetor de 10 elementos Float
```

#### Matrix

Representa uma estrutura bidimensional de elementos do mesmo tipo.

**Sintaxe:** `Matrix<tipo_elemento, linhas, colunas>`

**Exemplo:**
```omega
Variable m : Matrix<Int, 3, 4>;  // Matriz 3x4 de elementos Int
```
```

## 4. [P2] US-3.0: Iniciar Implementação do ΩTuner (Stretch Goal)

### Arquitetura Proposta

O ΩTuner deve ser projetado como um componente que:

1. Recebe uma representação de modelo (ΩLang ou IR)
2. Define um espaço de busca para hiperparâmetros
3. Executa estratégias de otimização
4. Retorna parâmetros otimizados

```
┌─────────────┐     ┌───────────────┐     ┌──────────────────┐
│ Modelo em   │────▶│   ΩTuner      │────▶│  Parâmetros      │
│ ΩLang ou IR │     │ (Otimização)  │     │  Otimizados      │
└─────────────┘     └───────┬───────┘     └──────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Estratégias  │
                    │de Otimização  │
                    │               │
                    └───────────────┘
```

### Abordagem Técnica Recomendada

**Interface Inicial**

```python
from typing import Dict, Any, List, Union, Callable
import numpy as np

class OmegaTuner:
    """
    Componente responsável por otimizar hiperparâmetros de modelos no sistema Omega.
    """
    
    def __init__(self):
        """Inicializa o ΩTuner."""
        self.optimization_strategies = {
            "grid_search": self._grid_search,
            "random_search": self._random_search,
            # Outras estratégias serão adicionadas no futuro
        }
    
    def tune(self, 
             model_representation: Dict[str, Any],
             parameter_space: Dict[str, Union[List, Dict]],
             objective_function: Callable,
             strategy: str = "grid_search",
             max_iterations: int = 100) -> Dict[str, Any]:
        """
        Otimiza os hiperparâmetros do modelo.
        
        Args:
            model_representation: Representação do modelo em ΩLang ou IR
            parameter_space: Espaço de busca para os parâmetros
                Ex: {"learning_rate": [0.001, 0.01, 0.1], "regularization": ["l1", "l2"]}
            objective_function: Função que avalia a qualidade do modelo
                Deve aceitar (model_representation, parameters) e retornar um score
            strategy: Estratégia de otimização a ser usada
            max_iterations: Número máximo de iterações
            
        Returns:
            Dicionário com os parâmetros otimizados e metadados da otimização
        """
        if strategy not in self.optimization_strategies:
            raise ValueError(f"Estratégia de otimização desconhecida: {strategy}")
            
        optimization_func = self.optimization_strategies[strategy]
        return optimization_func(model_representation, parameter_space, objective_function, max_iterations)
    
    def _grid_search(self, model, parameter_space, objective_function, max_iterations):
        """Implementa busca em grade para otimização."""
        # Implementação básica de busca em grade
        # ...
        
    def _random_search(self, model, parameter_space, objective_function, max_iterations):
        """Implementa busca aleatória para otimização."""
        # Implementação básica de busca aleatória
        # ...
```

**Pesquisa de Bibliotecas de Otimização**

Avaliar as seguintes bibliotecas para possível integração:

1. **Optuna**
   - Prós: API flexível, visualizações integradas, suporte a pruning
   - Contras: Dependência adicional

2. **Hyperopt**
   - Prós: Algoritmos bayesianos, distribuições complexas
   - Contras: API menos intuitiva

3. **Ray Tune**
   - Prós: Paralelização eficiente, integração com frameworks de ML
   - Contras: Overhead para casos simples

4. **Scikit-optimize**
   - Prós: Integração com scikit-learn, API simples
   - Contras: Menos recursos avançados

**Integração com a Equação Omega**

Definir funções objetivo que incorporem a equação Omega:

```python
def omega_objective(model_params, data, weights):
    """
    Calcula o valor da função objetivo Omega.
    
    Omega = S + λC - βA + μE
    
    Args:
        model_params: Parâmetros do modelo a serem avaliados
        data: Dados para avaliação
        weights: Pesos λ, β, μ para os termos da equação
        
    Returns:
        Valor da função objetivo (menor é melhor)
    """
    # Treinar modelo com os parâmetros
    model = train_model(data, model_params)
    
    # Calcular termos da equação
    S = calculate_surprise(model, data)  # Entropia/imprevisibilidade
    C = calculate_complexity(model)      # Complexidade do modelo
    A = calculate_adaptation(model, data) # Melhoria no ambiente
    E = calculate_energy(model)          # Custo energético/computacional
    
    # Aplicar pesos
    lambda_weight = weights.get('lambda', 1.0)
    beta_weight = weights.get('beta', 1.0)
    mu_weight = weights.get('mu', 1.0)
    
    # Calcular Omega (queremos minimizar)
    omega = S + lambda_weight * C - beta_weight * A + mu_weight * E
    
    return omega
```

### Implementação Inicial Recomendada

Para o Sprint 3, focar em:

1. Definir a interface clara do ΩTuner
2. Implementar uma estratégia simples (busca em grade)
3. Criar documentação sobre as bibliotecas avaliadas
4. Preparar a integração com a equação Omega

A implementação completa com estratégias avançadas pode ser deixada para sprints futuros.
