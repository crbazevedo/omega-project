# Documentação das Extensões de Sintaxe - Sprint 5

Este documento detalha as extensões de sintaxe implementadas no parser ΩMetaLang durante o Sprint 5, como parte da reestruturação arquitetural.

## 1. Tipos Genéricos

O parser agora suporta a definição e o uso de tipos genéricos, permitindo maior flexibilidade e expressividade na declaração de variáveis e estruturas de dados.

### Sintaxe

A sintaxe para tipos genéricos segue o padrão `NomeTipo<ParametroTipo1, ParametroTipo2, ...>`.

**Tipos Parametrizados Pré-definidos:**

*   `Vector<TipoElemento>`: Vetor unidimensional. O tamanho pode ser opcionalmente especificado após o tipo do elemento (ex: `Vector<Float, 10>`).
*   `Matrix<TipoElemento, Linhas, Colunas>`: Matriz bidimensional. As dimensões são opcionais.
*   `Tensor<TipoElemento, Dim1, Dim2, ...>`: Tensor multidimensional. As dimensões são opcionais.
*   `Distribution<TipoElemento>`: Representa uma distribuição de probabilidade sobre um tipo específico.
*   `Space<TipoElemento>`: Representa um espaço (discreto ou contínuo) sobre um tipo específico.

**Tipos Genéricos Definidos pelo Usuário:**

*   `NomeTipoCustomizado<ParametroTipo1, ParametroTipo2, ...>`: Permite referenciar tipos genéricos definidos pelo usuário ou presentes na ΩIntelligenceBase.

**Tipos de Distribuição e Espaço Pré-definidos (Nomes Específicos):**

*   `Distribution<Normal>`
*   `Distribution<Uniform>`
*   `Distribution<Categorical>`
*   `Distribution<Bernoulli>`
*   `Space<Discrete>`
*   `Space<Continuous>`
*   `Space<Box>`

### Implementação

*   **Gramática (`grammar_modules/types.py`):** Foram adicionadas/modificadas as regras `vector_type`, `matrix_type`, `tensor_type`, `distribution_type`, `space_type`, `generic_type`, `type_param_list`, `predefined_distribution_type`, `predefined_space_type`, `distribution_type_name`, e `space_type_name` para acomodar a nova sintaxe.
*   **Transformer (`transformer.py`):** Foram adicionados/modificados os métodos `vector_type`, `matrix_type`, `tensor_type`, `distribution_type`, `space_type`, `generic_type`, `type_param_list`, `predefined_distribution_type`, `predefined_space_type`, `distribution_type_name`, e `space_type_name` para processar essas regras e construir a representação da AST (agora usando objetos `Tree` do Lark).

### Exemplo

```omega
OmegaSpec ExemploTiposGenericos {
    Variable meuVetor: Vector<Float>;
    Variable minhaMatriz: Matrix<Int, 3, 3>;
    Variable meuTensor: Tensor<Bool, 2, 4, 8>;
    Variable minhaDist: Distribution<Normal>;
    Variable meuEspaco: Space<Discrete>;
    Variable meuTipoCustom: ModeloGenerico<String, Int>;
    Variable vetorDeDist: Vector<Distribution<Float>>;

    Objective {
        Minimize: 0;
    }
}
```

## 2. Imports e Módulos

Foi adicionado suporte preliminar para declarações de importação, permitindo que arquivos ΩMetaLang referenciem outros módulos ou componentes.

### Sintaxe

*   `Import caminho.do.modulo;`: Importa um módulo inteiro.
*   `Import caminho.do.modulo.{ item1, item2, ClasseA };`: Importa itens específicos de um módulo.

### Implementação

*   **Gramática (`grammar_modules/imports.py`):** Foi criado um novo módulo de gramática para definir as regras `import_statement`, `import_target`, `module_path`, e `import_list`.
*   **Gramática Principal (`grammar_modules/program.py`):** A regra `program` foi atualizada para incluir `imports` antes das `declarations`.
*   **Transformer (`transformer.py`):** Foram adicionados os métodos `import_statement`, `import_target`, `module_path`, e `import_list` para processar as declarações de import.

### Exemplo

```omega
OmegaSpec ExemploImports {
    Import meu.pacote.utils;
    Import outro.modulo.{ FuncaoUtil, ClasseModelo };

    Variable x: utils.TipoUtil;
    Variable y: ClasseModelo;

    Objective {
        Minimize: FuncaoUtil(x, y);
    }
}
```

**Observação:** A integração completa e a resolução de imports no ΩKernelRuntime ainda precisam ser implementadas.

## 3. Comentários Multilinha

O parser agora ignora corretamente comentários de bloco no estilo C (`/* ... */`), além dos comentários de linha única (`// ...`) que já eram suportados.

### Sintaxe

```omega
/*
Este é um comentário
que abrange múltiplas
linhas.
*/
Variable x: Int; // Comentário de linha única
```

### Implementação

*   **Gramática (`grammar_modules/grammar.py`):** A diretiva `%ignore /\/\*(.|\n)*?\*\//` foi adicionada aos `COMMON_IMPORTS` para instruir o lexer a ignorar os blocos de comentários multilinha.

Nenhuma alteração foi necessária no transformer, pois os comentários são descartados antes da fase de parsing.

