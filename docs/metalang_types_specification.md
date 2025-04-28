# Especificação Detalhada dos Tipos ΩMetaLang

Este documento fornece uma especificação técnica detalhada do sistema de tipos do ΩMetaLang, complementando o Guia de Referência geral. Aqui, exploramos em profundidade cada tipo disponível, suas propriedades, operações permitidas e exemplos de uso.

## Tipos Primitivos

### Int

Representa números inteiros com precisão arbitrária.

**Características:**
- Faixa: Limitada apenas pela memória disponível
- Operações: `+`, `-`, `*`, `/` (divisão inteira), `%` (módulo), `^` (potência)
- Comparações: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Conversões: `Int -> Float`, `Int -> String`

**Exemplos:**
```
var a: Int = 42;
var b: Int = -7;
var c: Int = a + b;  // c = 35
var d: Int = a / b;  // d = -6 (divisão inteira)
var e: Int = a % b;  // e = 0
var f: Int = a ^ 2;  // f = 1764
```

### Float

Representa números de ponto flutuante de precisão dupla (IEEE 754).

**Características:**
- Precisão: 15-17 dígitos significativos
- Operações: `+`, `-`, `*`, `/`, `%`, `^`
- Comparações: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Conversões: `Float -> Int` (truncamento), `Float -> String`
- Funções especiais: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, etc.

**Exemplos:**
```
var a: Float = 3.14159;
var b: Float = -2.5;
var c: Float = a + b;      // c = 0.64159
var d: Float = a / b;      // d = -1.256636
var e: Float = sin(a);     // e = 0.00159...
var f: Float = sqrt(abs(b)); // f = 1.58113...
```

### Bool

Representa valores lógicos verdadeiro ou falso.

**Características:**
- Valores: `true`, `false`
- Operações: `&&` (AND), `||` (OR), `!` (NOT)
- Comparações: `==`, `!=`
- Conversões: `Bool -> Int` (1 para true, 0 para false), `Bool -> String`

**Exemplos:**
```
var a: Bool = true;
var b: Bool = false;
var c: Bool = a && b;  // c = false
var d: Bool = a || b;  // d = true
var e: Bool = !a;      // e = false
var f: Bool = (5 > 3); // f = true
```

### String

Representa sequências de caracteres Unicode.

**Características:**
- Codificação: UTF-8
- Operações: `+` (concatenação)
- Comparações: `==`, `!=`, `<`, `>`, `<=`, `>=` (ordem lexicográfica)
- Métodos: `length()`, `substring(start, end)`, `contains(substr)`, etc.

**Exemplos:**
```
var a: String = "Olá";
var b: String = "mundo";
var c: String = a + " " + b;  // c = "Olá mundo"
var d: Bool = a == b;         // d = false
var e: Int = a.length();      // e = 3
var f: Bool = c.contains(a);  // f = true
```

## Tipos Compostos

### Vector\<T\>

Representa uma coleção ordenada unidimensional de elementos do mesmo tipo.

**Características:**
- Tipo genérico: `T` pode ser qualquer tipo válido
- Indexação: Baseada em zero
- Operações: Acesso por índice, concatenação, fatiamento
- Métodos: `length()`, `append(item)`, `remove(index)`, etc.

**Exemplos:**
```
var a: Vector<Int> = [1, 2, 3, 4, 5];
var b: Vector<String> = ["a", "b", "c"];
var c: Int = a[2];                 // c = 3
var d: Vector<Int> = a[1:3];       // d = [2, 3]
var e: Int = a.length();           // e = 5
a.append(6);                       // a = [1, 2, 3, 4, 5, 6]
a.remove(0);                       // a = [2, 3, 4, 5, 6]

// Operações vetoriais
var v1: Vector<Float> = [1.0, 2.0, 3.0];
var v2: Vector<Float> = [4.0, 5.0, 6.0];
var v3: Vector<Float> = v1 + v2;   // v3 = [5.0, 7.0, 9.0]
var v4: Vector<Float> = v1 * 2.0;  // v4 = [2.0, 4.0, 6.0]
var dot: Float = v1 • v2;          // dot = 32.0 (produto escalar)
```

### Matrix\<T\>

Representa uma matriz bidimensional de elementos do mesmo tipo.

**Características:**
- Tipo genérico: `T` pode ser qualquer tipo válido
- Indexação: Baseada em zero, formato [linha, coluna]
- Operações: Acesso por índice, operações matriciais (soma, multiplicação, etc.)
- Métodos: `rows()`, `cols()`, `transpose()`, `determinant()` (para tipos numéricos), etc.

**Exemplos:**
```
var a: Matrix<Float> = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
];
var b: Float = a[0, 1];           // b = 2.0
var c: Vector<Float> = a[1];      // c = [4.0, 5.0, 6.0] (segunda linha)
var d: Int = a.rows();            // d = 2
var e: Int = a.cols();            // e = 3
var f: Matrix<Float> = a.transpose(); // f = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

// Operações matriciais
var m1: Matrix<Float> = [[1.0, 2.0], [3.0, 4.0]];
var m2: Matrix<Float> = [[5.0, 6.0], [7.0, 8.0]];
var m3: Matrix<Float> = m1 + m2;   // m3 = [[6.0, 8.0], [10.0, 12.0]]
var m4: Matrix<Float> = m1 * m2;   // m4 = [[19.0, 22.0], [43.0, 50.0]] (multiplicação matricial)
var det: Float = m1.determinant(); // det = -2.0
```

### Tensor\<T\>

Representa um tensor multidimensional de elementos do mesmo tipo.

**Características:**
- Tipo genérico: `T` pode ser qualquer tipo válido
- Dimensões: Arbitrárias (1D = vetor, 2D = matriz, 3D+)
- Indexação: Baseada em zero, formato [dim1, dim2, ...]
- Operações: Acesso por índice, operações tensoriais
- Métodos: `shape()`, `reshape(dims)`, `slice(start, end)`, etc.

**Exemplos:**
```
var a: Tensor<Float> = [
    [[1.0, 2.0], [3.0, 4.0]],
    [[5.0, 6.0], [7.0, 8.0]]
];  // Tensor 2x2x2
var b: Float = a[0, 1, 0];        // b = 3.0
var c: Vector<Int> = a.shape();   // c = [2, 2, 2]
var d: Tensor<Float> = a.reshape([4, 2]); // d = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
var e: Tensor<Float> = a.slice([0, 0, 0], [1, 2, 2]); // e = [[[1.0, 2.0], [3.0, 4.0]]]

// Operações tensoriais
var t1: Tensor<Float> = a * 2.0;  // Multiplicação escalar
var t2: Tensor<Float> = a + a;    // Soma elemento a elemento
```

### Map\<K, V\>

Representa um mapeamento de chaves para valores.

**Características:**
- Tipos genéricos: `K` (chave) e `V` (valor) podem ser quaisquer tipos válidos
- Chaves: Devem ser únicas e comparáveis
- Operações: Acesso por chave, inserção, remoção
- Métodos: `keys()`, `values()`, `contains(key)`, etc.

**Exemplos:**
```
var a: Map<String, Int> = {
    "um": 1,
    "dois": 2,
    "três": 3
};
var b: Int = a["dois"];           // b = 2
var c: Bool = a.contains("quatro"); // c = false
a["quatro"] = 4;                  // Adiciona novo par chave-valor
a.remove("um");                   // Remove par chave-valor
var d: Vector<String> = a.keys(); // d = ["dois", "três", "quatro"]
var e: Vector<Int> = a.values();  // e = [2, 3, 4]
```

### Option\<T\>

Representa um valor que pode estar presente ou ausente.

**Características:**
- Tipo genérico: `T` pode ser qualquer tipo válido
- Valores: `Some(valor)` ou `None`
- Métodos: `isSome()`, `isNone()`, `unwrap()`, `unwrapOr(default)`, etc.

**Exemplos:**
```
var a: Option<Int> = Some(42);
var b: Option<Int> = None;
var c: Bool = a.isSome();         // c = true
var d: Bool = b.isNone();         // d = true
var e: Int = a.unwrap();          // e = 42
var f: Int = b.unwrapOr(0);       // f = 0

// Uso com funções que podem falhar
function divideIfPossible(x: Int, y: Int) -> Option<Float> {
    if (y == 0) {
        return None;
    } else {
        return Some(x / y);
    }
}

var result: Option<Float> = divideIfPossible(10, 2);  // result = Some(5.0)
var error: Option<Float> = divideIfPossible(10, 0);   // error = None
```

### Tuple\<T1, T2, ...\>

Representa uma coleção ordenada de valores de tipos potencialmente diferentes.

**Características:**
- Tipos genéricos: `T1`, `T2`, etc. podem ser quaisquer tipos válidos
- Tamanho: Fixo na declaração
- Indexação: Baseada em zero
- Imutabilidade: Valores não podem ser alterados após a criação

**Exemplos:**
```
var a: Tuple<Int, String, Bool> = (42, "texto", true);
var b: Int = a[0];                // b = 42
var c: String = a[1];             // c = "texto"
var d: Bool = a[2];               // d = true

// Desestruturação
var (num, text, flag) = a;        // num = 42, text = "texto", flag = true

// Uso em funções que retornam múltiplos valores
function divideWithRemainder(x: Int, y: Int) -> Tuple<Int, Int> {
    return (x / y, x % y);
}

var (quotient, remainder) = divideWithRemainder(10, 3);  // quotient = 3, remainder = 1
```

## Tipos Especiais

### Distribution\<T\>

Representa uma distribuição de probabilidade sobre valores do tipo T.

**Características:**
- Tipo genérico: `T` pode ser qualquer tipo válido
- Operações: Amostragem, cálculo de probabilidades, entropia
- Métodos: `sample()`, `pdf(value)` ou `pmf(value)`, `cdf(value)`, `entropy()`, etc.

**Exemplos:**
```
var a: Distribution<Float> = normal(0.0, 1.0);  // Distribuição normal padrão
var b: Float = a.sample();                      // Amostra aleatória da distribuição
var c: Float = a.pdf(0.5);                      // Densidade de probabilidade em x=0.5
var d: Float = a.cdf(1.0);                      // Probabilidade acumulada até x=1.0
var e: Float = a.entropy();                     // Entropia da distribuição

var f: Distribution<Int> = poisson(5.0);        // Distribuição de Poisson com lambda=5
var g: Float = f.pmf(3);                        // Massa de probabilidade para x=3

var h: Distribution<Bool> = bernoulli(0.7);     // Distribuição de Bernoulli com p=0.7
var i: Vector<Bool> = h.sample(10);             // Vetor com 10 amostras
```

### Model\<I, O\>

Representa um modelo que mapeia entradas do tipo I para saídas do tipo O.

**Características:**
- Tipos genéricos: `I` (entrada) e `O` (saída) podem ser quaisquer tipos válidos
- Parâmetros: Variáveis treináveis que definem o comportamento do modelo
- Métodos: `forward(input)`, `predict(input)`, `train(data)`, etc.

**Exemplos:**
```
var a: Model<Vector<Float>, Int> = LogisticRegression {
    weights: initialize.xavier(10, 3),
    bias: initialize.zeros(3)
};

var input: Vector<Float> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
var output: Int = a.predict(input);  // Classificação (0, 1 ou 2)

var b: Model<Matrix<Float>, Vector<Float>> = LinearRegression {
    weights: initialize.normal(5, 1),
    bias: 0.0
};

var features: Matrix<Float> = loadData("features.csv");
var targets: Vector<Float> = loadData("targets.csv");
b.train(features, targets);
var predictions: Vector<Float> = b.predict(features);
```

### Environment\<S, A\>

Representa um ambiente com estados do tipo S e ações do tipo A.

**Características:**
- Tipos genéricos: `S` (estado) e `A` (ação) podem ser quaisquer tipos válidos
- Métodos: `reset()`, `step(action)`, `render()`, etc.

**Exemplos:**
```
var a: Environment<Vector<Float>, Int> = CartPole {
    gravity: 9.8,
    pole_length: 0.5,
    max_steps: 500
};

var state: Vector<Float> = a.reset();  // Estado inicial
var action: Int = 1;                   // Mover para a direita
var result: Tuple<Vector<Float>, Float, Bool> = a.step(action);  // (novo_estado, recompensa, terminado)
var (new_state, reward, done) = result;

a.render();  // Visualização do ambiente
```

### Agent\<S, A\>

Representa um agente que observa estados do tipo S e executa ações do tipo A.

**Características:**
- Tipos genéricos: `S` (estado) e `A` (ação) podem ser quaisquer tipos válidos
- Política: Estratégia para escolher ações com base em estados
- Métodos: `act(state)`, `learn(state, action, reward, next_state)`, etc.

**Exemplos:**
```
var a: Agent<Vector<Float>, Int> = DQNAgent {
    state_dim: 4,
    action_dim: 2,
    learning_rate: 0.001,
    discount_factor: 0.99,
    exploration_rate: 0.1
};

var state: Vector<Float> = [0.1, 0.2, -0.3, 0.5];
var action: Int = a.act(state);  // Escolhe uma ação com base no estado atual

// Após executar a ação no ambiente e observar o resultado
var next_state: Vector<Float> = [0.12, 0.18, -0.25, 0.48];
var reward: Float = 1.0;
var done: Bool = false;
a.learn(state, action, reward, next_state, done);  // Atualiza a política com base na experiência
```

### Loss

Representa uma função de perda para otimização.

**Características:**
- Métodos: `compute(predictions, targets)`, `gradient(predictions, targets)`, etc.
- Tipos comuns: MSE, CrossEntropy, Hinge, etc.

**Exemplos:**
```
var a: Loss = MSE();
var predictions: Vector<Float> = [1.2, 2.3, 3.1];
var targets: Vector<Float> = [1.0, 2.0, 3.0];
var loss_value: Float = a.compute(predictions, targets);  // Valor da perda
var gradients: Vector<Float> = a.gradient(predictions, targets);  // Gradientes para backpropagation

var b: Loss = CrossEntropy();
var class_probs: Matrix<Float> = [
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.3, 0.5]
];
var true_classes: Vector<Int> = [0, 1, 2];
var ce_loss: Float = b.compute(class_probs, true_classes);
```

### Metric

Representa uma métrica para avaliação de desempenho.

**Características:**
- Métodos: `compute(predictions, targets)`, `update(predictions, targets)`, `reset()`, etc.
- Tipos comuns: Accuracy, F1Score, RMSE, etc.

**Exemplos:**
```
var a: Metric = Accuracy();
var predictions: Vector<Int> = [1, 0, 1, 1, 0];
var targets: Vector<Int> = [1, 0, 0, 1, 1];
var accuracy: Float = a.compute(predictions, targets);  // accuracy = 0.6

var b: Metric = F1Score();
b.update(predictions, targets);  // Acumula resultados
var more_predictions: Vector<Int> = [1, 1, 0];
var more_targets: Vector<Int> = [1, 0, 0];
b.update(more_predictions, more_targets);  // Acumula mais resultados
var f1: Float = b.compute();  // Calcula F1 com base em todos os dados acumulados
b.reset();  // Limpa dados acumulados
```

## Regras de Tipagem

### Inferência de Tipos

ΩMetaLang suporta inferência de tipos em muitos contextos, permitindo omitir anotações de tipo quando o tipo pode ser deduzido:

```
var a = 42;              // Inferido como Int
var b = 3.14;            // Inferido como Float
var c = true;            // Inferido como Bool
var d = "texto";         // Inferido como String
var e = [1, 2, 3];       // Inferido como Vector<Int>
var f = {"a": 1, "b": 2}; // Inferido como Map<String, Int>
```

### Conversão de Tipos

ΩMetaLang suporta conversões explícitas entre tipos compatíveis:

```
var a: Int = 42;
var b: Float = Float(a);      // b = 42.0
var c: String = String(a);    // c = "42"

var d: Float = 3.14;
var e: Int = Int(d);          // e = 3 (truncado)
var f: String = String(d);    // f = "3.14"

var g: String = "123";
var h: Int = Int(g);          // h = 123
var i: Float = Float(g);      // i = 123.0
```

### Verificação de Tipos

ΩMetaLang é uma linguagem estaticamente tipada, com verificação de tipos em tempo de compilação:

```
var a: Int = 42;
var b: String = a;  // Erro: não pode atribuir Int a String

function add(x: Int, y: Int) -> Int {
    return x + y;
}

add(1, 2);    // OK
add(1, "2");  // Erro: esperava Int, recebeu String
```

### Tipos Genéricos

ΩMetaLang suporta definição de funções e estruturas genéricas:

```
function identity<T>(x: T) -> T {
    return x;
}

var a: Int = identity(42);        // a = 42
var b: String = identity("texto"); // b = "texto"

function first<T, U>(pair: Tuple<T, U>) -> T {
    return pair[0];
}

var c: Int = first((10, "abc"));  // c = 10
```

## Considerações de Implementação

### Representação em Memória

- **Int**: Representado como inteiro de precisão arbitrária
- **Float**: Representado como IEEE 754 de precisão dupla (64 bits)
- **Bool**: Representado como um único bit (otimizado para armazenamento)
- **String**: Representado como sequência de caracteres UTF-8
- **Vector\<T\>**: Representado como array contíguo de elementos do tipo T
- **Matrix\<T\>**: Representado como array bidimensional ou array unidimensional linearizado
- **Tensor\<T\>**: Representado como array multidimensional ou array unidimensional linearizado com strides
- **Map\<K, V\>**: Representado como tabela hash ou árvore balanceada
- **Option\<T\>**: Representado como tag (Some/None) + valor (quando Some)
- **Tuple\<T1, T2, ...>**: Representado como valores contíguos em memória

### Otimizações

- **Especialização de Tipos**: Implementações otimizadas para tipos numéricos comuns
- **Lazy Evaluation**: Avaliação preguiçosa para expressões complexas
- **Copy-on-Write**: Semântica de cópia eficiente para tipos grandes
- **Inlining**: Expansão inline de funções pequenas
- **Vectorização**: Uso de instruções SIMD para operações em vetores e matrizes

## Extensibilidade

ΩMetaLang permite a definição de tipos personalizados através de estruturas e enumerações:

```
struct Point {
    x: Float;
    y: Float;
    
    function distance(other: Point) -> Float {
        return sqrt((x - other.x)^2 + (y - other.y)^2);
    }
}

var p1: Point = Point{x: 1.0, y: 2.0};
var p2: Point = Point{x: 4.0, y: 6.0};
var d: Float = p1.distance(p2);  // d = 5.0

enum Color {
    Red,
    Green,
    Blue,
    Custom(Int, Int, Int)
}

var c1: Color = Color.Red;
var c2: Color = Color.Custom(128, 0, 255);
```

---

*Este documento é parte da documentação oficial do Projeto Omega e complementa o Guia de Referência ΩMetaLang.*
