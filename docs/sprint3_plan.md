# Plano do Sprint 3 - Projeto Omega

## 1. Visão Geral

**Período:** 29/04/2025 a 13/05/2025 (2 semanas)

**Tema do Sprint:** Estabilização do Pipeline e Busca Inteligente

**Objetivo Principal:** Estabilizar o pipeline de processamento inicial (Parser -> Architect -> Synthesizer) e adicionar a capacidade de busca de artefatos, além de melhorar a documentação da linguagem core.

**Capacidade da Equipe:** 40 story points (estimativa baseada na velocidade dos sprints anteriores)

## 2. Prioridades e User Stories

### [P0 - Crítico] US-DT-1.1: Finalizar Correção do Parser ΩMetaLang (10 SP)

**Descrição:** Resolver o erro pendente no transformer Lark e garantir que todos os testes do parser passem, estabilizando a geração da AST.

**Critérios de Aceitação:**
- Todos os testes unitários do parser passam sem erros
- A AST gerada para os 10 scripts de exemplo é correta, especialmente na representação de tipos
- O parser lida corretamente com tipos parametrizados (ex: `Vector<Float, 5>`)

**Responsável Sugerido:** Desenvolvedor Core com experiência em parsers/Lark

### [P1 - Alto] US-2.2: Implementar ΩSearch para Recuperação de Artefatos (13 SP)

**Descrição:** Criar a funcionalidade básica de busca na ΩIntelligenceBase (mock) baseada em similaridade de embeddings.

**Critérios de Aceitação:**
- ΩSearch aceita uma representação de embedding como consulta
- ΩSearch interage com uma versão mock da ΩIntelligenceBase contendo artefatos de exemplo
- ΩSearch retorna uma lista ordenada dos k artefatos mais similares com seus respectivos scores
- A implementação usa uma métrica de similaridade adequada (ex: distância de cosseno)

**Responsável Sugerido:** Engenheiro de Dados / ML

### [P1 - Alto] DT-4.2: Documentar Guia de Referência ΩMetaLang (8 SP)

**Descrição:** Criar um guia de referência abrangente para a linguagem ΩMetaLang.

**Critérios de Aceitação:**
- Documentação cobre todos os aspectos da linguagem (sintaxe, tipos, declarações, expressões, objetivo)
- Exemplos claros para cada construção da linguagem
- Exemplos completos de código ΩMetaLang que demonstram o uso das diferentes funcionalidades
- Documento bem formatado, claro e correto

**Responsável Sugerido:** Documentação Técnica / Desenvolvedor Core

### [P2 - Médio] US-3.0: Iniciar Implementação do ΩTuner (Stretch Goal) (8 SP)

**Descrição:** Criar o esboço inicial do componente ΩTuner e pesquisar abordagens de otimização.

**Critérios de Aceitação:**
- Interface inicial da classe `OmegaTuner` definida
- Documentação de abordagens/bibliotecas para otimização de hiperparâmetros
- (Opcional) Implementação básica de uma estratégia de tuning para um caso de teste simples

**Responsável Sugerido:** Engenheiro de ML / Otimização

## 3. Tarefas Detalhadas

Ver documento `sprint3_refined_tasks.md` para o detalhamento completo de cada tarefa.

## 4. Dependências e Riscos

### Dependências
- A correção do parser (US-DT-1.1) é um pré-requisito para testes de integração confiáveis
- A implementação do ΩTuner (US-3.0) depende parcialmente da estabilidade do pipeline anterior

### Riscos
1. **Risco Alto:** A correção do parser pode ser mais complexa do que o esperado, especialmente se envolver limitações da biblioteca Lark
   - **Mitigação:** Considerar abordagens alternativas, como refatoração mais ampla do transformer ou até mesmo a substituição do Lark por outra biblioteca de parsing
   
2. **Risco Médio:** A implementação do ΩSearch pode ser desafiadora sem uma ΩIntelligenceBase real
   - **Mitigação:** Criar uma versão mock robusta da ΩIntelligenceBase com dados representativos

3. **Risco Baixo:** Falta de tempo para implementar o ΩTuner (stretch goal)
   - **Mitigação:** Priorizar as tarefas P0 e P1, deixando o ΩTuner como último item a ser abordado

## 5. Abordagem Técnica

### US-DT-1.1: Correção do Parser
- Investigar a fundo a documentação do Lark sobre transformers e o uso de `@v_args`
- Considerar a refatoração do transformer para usar uma abordagem mais simples e direta
- Focar na correção do método `simple_type()` e sua interação com a gramática

### US-2.2: ΩSearch
- Utilizar vetores de embedding pré-computados para a versão mock
- Implementar a similaridade de cosseno usando NumPy/SciPy para eficiência
- Estruturar a API para futura integração com sistemas de busca vetorial mais avançados (FAISS, Annoy)

### DT-4.2: Documentação ΩMetaLang
- Seguir o estilo de documentação de linguagens estabelecidas (Python, Rust)
- Usar Markdown com formatação rica (tabelas, blocos de código, etc.)
- Incluir diagramas de sintaxe para construções complexas

### US-3.0: ΩTuner
- Pesquisar bibliotecas como Optuna, Hyperopt e Ray Tune
- Focar inicialmente em uma interface clara e extensível
- Considerar a integração com a equação Omega (S + λC - βA + μE) para guiar a otimização

## 6. Métricas e Definição de Pronto

### Métricas de Sprint
- Número de testes unitários passando
- Cobertura de código dos novos componentes
- Qualidade da documentação (completude, clareza)

### Definição de Pronto para User Stories
- Código implementado e testado
- Documentação atualizada
- Code review concluído
- Testes unitários passando
- Branch mergeada para a main

## 7. Reuniões e Cerimônias

- **Sprint Planning:** 29/04/2025 (09:00-11:00)
- **Daily Standups:** Diariamente (09:30-09:45)
- **Sprint Review:** 13/05/2025 (14:00-15:00)
- **Sprint Retrospective:** 13/05/2025 (15:30-16:30)

## 8. Próximos Passos

1. Registrar os débitos técnicos como issues no GitHub
2. Refinar as estimativas de story points com a equipe
3. Atribuir responsáveis para cada tarefa
4. Iniciar o sprint com o foco nas tarefas P0
