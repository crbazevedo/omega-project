# Limitações do Parser ΩMetaLang Atual

## Resumo Executivo

Este documento identifica as limitações do parser ΩMetaLang atual em relação aos exemplos avançados implementados no Sprint 4. Estas limitações representam lacunas entre a expressividade da linguagem ΩMetaLang conforme demonstrada nos exemplos e a capacidade atual do parser de processar essa sintaxe.

## Limitações Identificadas

### 1. Problemas Estruturais

- **Exposição de Componentes**: O parser atual não expõe corretamente o `omega_grammar` como um componente importável, impedindo a reutilização da gramática em scripts de teste.
- **Integração entre Módulos**: Falta uma interface clara entre o parser e outros componentes do sistema, dificultando testes e extensões.

### 2. Limitações de Sintaxe

- **Tipos Complexos**: O parser atual não suporta adequadamente tipos complexos como `Distribution<T>`, `Environment<S,A>` e tipos aninhados.
- **Declarações Causais**: Não há suporte para a sintaxe `causal_model` e estruturas de grafo causal.
- **Garantias Formais**: A sintaxe `guarantees:` para especificação de propriedades formais não é reconhecida.
- **Operadores Especiais**: Operadores como `do()` para inferência causal não são suportados.

### 3. Limitações Semânticas

- **Verificação de Tipos**: O sistema atual não realiza verificação adequada de tipos para expressões complexas.
- **Resolução de Escopo**: Problemas na resolução de nomes e escopos, especialmente em contextos aninhados.
- **Transformação de AST**: Falhas na transformação da árvore sintática para representações intermediárias.

### 4. Problemas de Robustez

- **Tratamento de Erros**: Mensagens de erro pouco informativas e recuperação limitada de erros sintáticos.
- **Consistência**: Comportamento inconsistente ao processar diferentes construções da linguagem.
- **Escalabilidade**: Dificuldades em processar arquivos grandes ou com estruturas profundamente aninhadas.

## Impacto nas Funcionalidades

| Funcionalidade | Status | Impacto |
|----------------|--------|---------|
| Aprendizado por Reforço | Não suportado | Impossibilidade de expressar ambientes, agentes e políticas |
| Distribuições de Probabilidade | Não suportado | Impossibilidade de modelar incerteza e inferência bayesiana |
| Inferência Causal | Não suportado | Impossibilidade de expressar modelos causais e intervenções |
| Garantias Formais | Não suportado | Impossibilidade de especificar e verificar propriedades formais |

## Conclusão

O parser atual apresenta limitações significativas que impedem o processamento dos exemplos avançados implementados no Sprint 4. Estas limitações não são apenas técnicas, mas também conceituais, indicando a necessidade de uma revisão abrangente da arquitetura do parser para suportar a expressividade completa da linguagem ΩMetaLang.

A próxima seção deste documento proporá melhorias específicas para abordar estas limitações.
