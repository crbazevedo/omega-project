# Sprint 3 - Tarefas Refinadas

Com base nas prioridades definidas no documento `sprint3_prioritization.md`, detalhamos as User Stories e Débitos Técnicos em tarefas menores e mais gerenciáveis para o Sprint 3.

## 1. [P0] US-DT-1.1: Finalizar Correção do Parser ΩMetaLang

**Objetivo:** Resolver o erro pendente no transformer Lark e garantir que todos os testes do parser passem, estabilizando a geração da AST.

*   **Tarefa 1.1.1:** Investigar a causa raiz do erro `TypeError: OmegaMetaLangTransformer.simple_type() missing 1 required positional argument: 'type_name'`.
    *   Analisar a gramática Lark (`metalang_grammar.lark`).
    *   Revisar a estrutura do `OmegaMetaLangTransformer` e o uso de `@v_args`.
    *   Consultar a documentação do Lark sobre transformers e passagem de argumentos.
    *   Executar o parser em modo debug (se possível) para inspecionar o estado.
*   **Tarefa 1.1.2:** Implementar a correção no método `simple_type` ou refatorar a abordagem do transformer para tipos.
    *   Considerar diferentes estratégias de transformer (ex: com/sem `@v_args(inline=True)`).
    *   Ajustar a gramática se necessário para simplificar o transformer.
*   **Tarefa 1.1.3:** Executar todos os testes unitários do parser (`tests/unit/test_sprint1.py`) e garantir que passem.
*   **Tarefa 1.1.4:** Validar manualmente a AST gerada para os 10 scripts de exemplo do Sprint 1 (`examples/metalang/*.omega`), focando na correção dos tipos.
*   **Tarefa 1.1.5:** Fazer commit e push das correções para uma nova branch `sprint3/parser-final-fix`.

## 2. [P1] US-2.2: Implementar ΩSearch para Recuperação de Artefatos

**Objetivo:** Criar a funcionalidade básica de busca na ΩIntelligenceBase (mock) baseada em similaridade de embeddings.

*   **Tarefa 2.2.1:** Definir e criar a estrutura mock da ΩIntelligenceBase.
    *   Criar diretório `data/mock_intelligence_base`.
    *   Criar arquivos YAML/JSON de exemplo representando artefatos (ex: `model_template_01.yaml`, `component_xyz.json`).
    *   Incluir campos para metadados e um campo `embedding` com um vetor numérico simulado (lista de floats).
*   **Tarefa 2.2.2:** Criar a estrutura de diretórios para o componente ΩSearch (`src/core/search`).
*   **Tarefa 2.2.3:** Implementar a classe `OmegaSearch` em `src/core/search/search.py`.
    *   Método `__init__` para carregar os artefatos mock da `mock_intelligence_base`.
    *   Método `search(query_embedding: List[float], k: int)`.
*   **Tarefa 2.2.4:** Implementar a lógica de busca por similaridade de cosseno.
    *   Usar `numpy` ou `scipy.spatial.distance.cosine` para calcular a similaridade entre o `query_embedding` e os embeddings dos artefatos.
    *   Retornar os `k` artefatos mais similares (menor distância de cosseno) com seus metadados e score de similaridade.
*   **Tarefa 2.2.5:** Criar testes unitários para `OmegaSearch` em `tests/unit/test_search.py`.
    *   Testar o carregamento dos artefatos mock.
    *   Testar a função `search` com embeddings de consulta conhecidos e verificar se os artefatos corretos são retornados na ordem esperada.
    *   Testar casos limite (k=0, k > número de artefatos).
*   **Tarefa 2.2.6:** Fazer commit e push da implementação e testes para a branch `sprint3/search-implementation`.

## 3. [P1] DT-4.2: Documentar Guia de Referência ΩMetaLang

**Objetivo:** Criar um guia de referência abrangente para a linguagem ΩMetaLang.

*   **Tarefa 4.2.1:** Revisar a gramática Lark (`metalang_grammar.lark`) e a implementação do parser (`metalang_parser.py`) para garantir que a documentação esteja alinhada com o código.
*   **Tarefa 4.2.2:** Criar o arquivo `docs/metalang_reference_guide.md` e definir sua estrutura (Introdução, Sintaxe, Tipos, Declarações, Expressões, Objetivo, Exemplos).
*   **Tarefa 4.2.3:** Escrever as seções de Introdução, Sintaxe Básica e Tipos de Dados, com exemplos claros para cada construção.
*   **Tarefa 4.2.4:** Escrever a seção de Declarações (Variable, Model, Domain, Environment, Action), detalhando a sintaxe e os campos permitidos para cada uma.
*   **Tarefa 4.2.5:** Escrever as seções de Expressões (Literais, Referências, Operadores, Chamadas de Função) e a seção de Objetivo (Minimize, Maximize, Constraints, Weights).
*   **Tarefa 4.2.6:** Adicionar exemplos completos de código ΩMetaLang que demonstrem o uso das diferentes funcionalidades da linguagem.
*   **Tarefa 4.2.7:** Revisar o documento completo quanto à clareza, correção e formatação.
*   **Tarefa 4.2.8:** Fazer commit e push da documentação para a branch `sprint3/metalang-docs`.

## 4. [P2] US-3.0: Iniciar Implementação do ΩTuner (Stretch Goal)

**Objetivo:** Criar o esboço inicial do componente ΩTuner e pesquisar abordagens de otimização.

*   **Tarefa 3.0.1:** Criar a estrutura de diretórios para o componente ΩTuner (`src/core/tuner`).
*   **Tarefa 3.0.2:** Definir a interface inicial da classe `OmegaTuner` em `src/core/tuner/tuner.py`.
    *   Decidir o input (ΩLang? IR? Configuração específica do modelo?).
    *   Decidir o output (Parâmetros otimizados? Configuração atualizada?).
*   **Tarefa 3.0.3:** Implementar um esboço da classe `OmegaTuner` com métodos placeholder (ex: `tune(target_representation)`).
*   **Tarefa 3.0.4:** Pesquisar e documentar brevemente abordagens/bibliotecas para otimização de hiperparâmetros (ex: Optuna, Hyperopt, Ray Tune, busca em grade/aleatória).
*   **Tarefa 3.0.5:** (Opcional) Se o tempo permitir, implementar uma lógica de tuning muito básica (ex: busca em grade simples para um parâmetro) para um caso de teste interno.
*   **Tarefa 3.0.6:** Fazer commit e push do esboço inicial e da pesquisa para a branch `sprint3/tuner-initial`.
