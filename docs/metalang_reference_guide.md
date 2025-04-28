# Guia de Referência ΩMetaLang

## Introdução

ΩMetaLang é a linguagem declarativa central do Projeto Omega, projetada para especificar problemas de inferência, aprendizado, planejamento e controle adaptativo de forma concisa e expressiva. Esta linguagem serve como a interface principal para usuários que desejam aproveitar o poder do sistema Omega sem precisar entender os detalhes de implementação subjacentes.

Este guia de referência documenta a sintaxe, tipos, declarações e expressões disponíveis no ΩMetaLang, fornecendo exemplos práticos para ilustrar padrões de uso comuns.

## Fundamentos Teóricos

ΩMetaLang é baseada na teoria unificadora Omega:

```
Ω = S + λC - βA + μE
```

Onde:
- **S** é a surpresa (entropia, imprevisibilidade)
- **C** é a complexidade (modelo, capacidade, custo estrutural)
- **A** é a ação adaptativa (melhoria no ambiente)
- **E** é o custo energético/computacional

Esta formulação matemática permite que o sistema Omega equilibre automaticamente estes fatores para encontrar soluções ótimas para uma ampla variedade de problemas.

## Sintaxe Básica

Um programa ΩMetaLang consiste em um cabeçalho seguido por uma série de declarações:

```
Omega NomeDoPrograma {
    // Declarações
}
```

Comentários podem ser incluídos usando `//` para comentários de linha única ou `/* ... */` para comentários de múltiplas linhas.

## Sistema de Tipos

ΩMetaLang possui um sistema de tipos rico que permite expressar uma variedade de estruturas de dados e relações:

### Tipos Primitivos

- **Int**: Números inteiros (ex: `42`, `-7`)
- **Float**: Números de ponto flutuante (ex: `3.14`, `-0.01`, `1.0e-5`)
- **Bool**: Valores booleanos (`true`, `false`)
- **String**: Sequências de caracteres (ex: `"texto"`)

### Tipos Compostos

- **Vector\<T\>**: Coleção ordenada de elementos do tipo T (ex: `Vector<Float>`)
- **Matrix\<T\>**: Matriz bidimensional de elementos do tipo T (ex: `Matrix<Float>`)
- **Tensor\<T\>**: Tensor multidimensional de elementos do tipo T (ex: `Tensor<Float>`)
- **Map\<K, V\>**: Mapeamento de chaves do tipo K para valores do tipo V (ex: `Map<String, Float>`)
- **Option\<T\>**: Valor opcional do tipo T (pode ser `Some(valor)` ou `None`)
- **Tuple\<T1, T2, ...\>**: Tupla de valores de tipos potencialmente diferentes (ex: `Tuple<Int, String>`)

### Tipos Especiais

- **Distribution\<T\>**: Distribuição de probabilidade sobre valores do tipo T (ex: `Distribution<Float>`)
- **Model\<I, O\>**: Modelo que mapeia entradas do tipo I para saídas do tipo O (ex: `Model<Vector<Float>, Int>`)
- **Environment\<S, A\>**: Ambiente com estados do tipo S e ações do tipo A
- **Agent\<S, A\>**: Agente que observa estados do tipo S e executa ações do tipo A
- **Loss**: Função de perda para otimização
- **Metric**: Métrica para avaliação de desempenho

## Declarações

### Declaração de Variáveis

```
var nome: Tipo = valor;
```

Exemplos:
```
var x: Int = 42;
var learning_rate: Float = 0.01;
var is_training: Bool = true;
var name: String = "modelo_classificacao";
```

### Declaração de Constantes

```
const nome: Tipo = valor;
```

Exemplo:
```
const PI: Float = 3.14159;
const MAX_ITERATIONS: Int = 1000;
```

### Declaração de Modelos

```
model NomeDoModelo<TipoEntrada, TipoSaída> {
    // Parâmetros e configuração
}
```

Exemplo:
```
model LogisticRegression<Vector<Float>, Float> {
    param weights: Vector<Float>;
    param bias: Float;
    
    config regularization: String = "l2";
    config learning_rate: Float = 0.01;
}
```

### Declaração de Dados

```
data NomeDoConjunto {
    // Especificação dos dados
}
```

Exemplo:
```
data TrainingSet {
    source: "path/to/data.csv";
    format: "csv";
    features: ["x1", "x2", "x3"];
    target: "y";
    split: {train: 0.8, validation: 0.1, test: 0.1};
}
```

### Declaração de Objetivos

```
objective NomeDoObjetivo {
    // Especificação do objetivo
}
```

Exemplo:
```
objective Classificacao {
    minimize: CrossEntropyLoss;
    constraints: [
        accuracy > 0.9,
        inference_time < 100ms
    ];
}
```

### Declaração de Experimentos

```
experiment NomeDoExperimento {
    // Configuração do experimento
}
```

Exemplo:
```
experiment HiperparametrosTuning {
    model: MyModel;
    data: MyDataset;
    objective: MyObjective;
    search_space: {
        learning_rate: range(0.001, 0.1, log=true),
        batch_size: [16, 32, 64, 128],
        dropout: range(0.1, 0.5)
    };
    search_algorithm: "bayesian";
    max_trials: 50;
}
```

## Expressões

### Expressões Aritméticas

ΩMetaLang suporta operações aritméticas padrão:

- Adição: `a + b`
- Subtração: `a - b`
- Multiplicação: `a * b`
- Divisão: `a / b`
- Módulo: `a % b`
- Potência: `a ^ b`

### Expressões Lógicas

- AND: `a && b`
- OR: `a || b`
- NOT: `!a`
- Igualdade: `a == b`
- Desigualdade: `a != b`
- Maior que: `a > b`
- Menor que: `a < b`
- Maior ou igual: `a >= b`
- Menor ou igual: `a <= b`

### Expressões Condicionais

```
if (condição) {
    // código se verdadeiro
} else {
    // código se falso
}
```

### Expressões de Iteração

```
for (var i: Int = 0; i < n; i = i + 1) {
    // código a ser repetido
}
```

```
while (condição) {
    // código a ser repetido
}
```

### Expressões Funcionais

```
map(vetor, x => x * 2)
filter(vetor, x => x > 0)
reduce(vetor, (acc, x) => acc + x, 0)
```

## Funções Integradas

ΩMetaLang fornece várias funções integradas para operações comuns:

### Funções Matemáticas

- `sin(x)`, `cos(x)`, `tan(x)`: Funções trigonométricas
- `exp(x)`: Função exponencial
- `log(x)`, `log10(x)`: Logaritmos natural e base 10
- `sqrt(x)`: Raiz quadrada
- `abs(x)`: Valor absoluto
- `floor(x)`, `ceil(x)`, `round(x)`: Arredondamento

### Funções Estatísticas

- `mean(x)`: Média aritmética
- `median(x)`: Mediana
- `std(x)`: Desvio padrão
- `var(x)`: Variância
- `corr(x, y)`: Correlação
- `cov(x, y)`: Covariância

### Funções de Distribuição

- `normal(mean, std)`: Distribuição normal
- `uniform(min, max)`: Distribuição uniforme
- `bernoulli(p)`: Distribuição de Bernoulli
- `categorical(probs)`: Distribuição categórica
- `sample(dist, n)`: Amostra n valores da distribuição

### Funções de Manipulação de Dados

- `shape(tensor)`: Retorna as dimensões de um tensor
- `reshape(tensor, shape)`: Reformata um tensor
- `concat(a, b)`: Concatena tensores
- `slice(tensor, start, end)`: Extrai uma fatia de um tensor
- `transpose(matrix)`: Transpõe uma matriz

## Exemplos Completos

### Exemplo 1: Regressão Logística

```
Omega LogisticRegressionExample {
    // Definição dos dados
    data IrisDataset {
        source: "datasets/iris.csv";
        features: ["sepal_length", "sepal_width", "petal_length", "petal_width"];
        target: "species";
        preprocess: {
            normalize: "standard",
            one_hot_encode: ["species"]
        };
    }
    
    // Definição do modelo
    model LogisticRegression<Vector<Float>, Vector<Float>> {
        param weights: Matrix<Float> = initialize.xavier(4, 3);
        param bias: Vector<Float> = initialize.zeros(3);
        
        config learning_rate: Float = 0.01;
        config regularization: Float = 0.001;
        config max_iterations: Int = 1000;
        
        forward(x: Vector<Float>) -> Vector<Float> {
            return softmax(x * weights + bias);
        }
    }
    
    // Definição do objetivo
    objective Classification {
        minimize: cross_entropy(model.forward(data.features), data.target);
        regularize: l2(model.weights, model.regularization);
        
        metrics: {
            accuracy: accuracy(model.forward(data.features), data.target),
            f1_score: f1_score(model.forward(data.features), data.target)
        };
    }
    
    // Definição do experimento
    experiment TrainAndEvaluate {
        model: LogisticRegression;
        data: IrisDataset;
        objective: Classification;
        
        optimizer: Adam {
            learning_rate: model.learning_rate,
            beta1: 0.9,
            beta2: 0.999
        };
        
        train: {
            batch_size: 32,
            epochs: 50,
            early_stopping: {
                monitor: "validation_loss",
                patience: 5
            }
        };
        
        evaluate: {
            metrics: ["accuracy", "f1_score", "confusion_matrix"]
        };
    }
}
```

### Exemplo 2: Rede Neural para Séries Temporais

```
Omega TimeSeriesForecastingExample {
    // Definição dos dados
    data StockPrices {
        source: "datasets/stock_prices.csv";
        time_column: "date";
        features: ["open", "high", "low", "volume"];
        target: "close";
        sequence_length: 30;
        horizon: 5;
        split: {train: 0.7, validation: 0.15, test: 0.15};
        preprocess: {
            normalize: "min_max",
            fill_missing: "forward"
        };
    }
    
    // Definição do modelo
    model LSTM<Tensor<Float>, Vector<Float>> {
        param input_size: Int = 4;
        param hidden_size: Int = 64;
        param num_layers: Int = 2;
        param output_size: Int = 5;
        
        config dropout: Float = 0.2;
        config learning_rate: Float = 0.001;
        
        architecture: {
            lstm_layers: [
                LSTM(input_size, hidden_size, dropout),
                LSTM(hidden_size, hidden_size, dropout)
            ],
            output_layer: Linear(hidden_size, output_size)
        };
    }
    
    // Definição do objetivo
    objective Forecasting {
        minimize: mse(model.forward(data.features), data.target);
        
        metrics: {
            mae: mae(model.forward(data.features), data.target),
            rmse: rmse(model.forward(data.features), data.target),
            mape: mape(model.forward(data.features), data.target)
        };
    }
    
    // Definição do experimento
    experiment HyperparameterTuning {
        model: LSTM;
        data: StockPrices;
        objective: Forecasting;
        
        search_space: {
            hidden_size: [32, 64, 128],
            num_layers: [1, 2, 3],
            dropout: range(0.1, 0.5, step=0.1),
            learning_rate: range(0.0001, 0.01, log=true)
        };
        
        search_algorithm: "random";
        max_trials: 20;
        
        train: {
            batch_size: 64,
            epochs: 100,
            early_stopping: {
                monitor: "validation_loss",
                patience: 10
            }
        };
    }
}
```

## Boas Práticas

1. **Nomeação**: Use nomes descritivos para variáveis, modelos e experimentos.
2. **Comentários**: Documente seu código com comentários explicativos.
3. **Modularidade**: Divida problemas complexos em componentes menores e reutilizáveis.
4. **Validação**: Sempre inclua métricas de validação para avaliar o desempenho do modelo.
5. **Reprodutibilidade**: Defina seeds para geradores de números aleatórios quando necessário.
6. **Escalabilidade**: Considere o custo computacional ao projetar modelos e experimentos.

## Referências

- Documentação completa do Projeto Omega
- Especificação formal da gramática ΩMetaLang
- Biblioteca de exemplos ΩMetaLang
- Tutoriais e guias práticos

---

*Este guia de referência é parte da documentação oficial do Projeto Omega.*
