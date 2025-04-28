# Especificação Detalhada das Declarações e Expressões ΩMetaLang

Este documento fornece uma especificação técnica detalhada das declarações e expressões disponíveis no ΩMetaLang, complementando o Guia de Referência geral e a Especificação de Tipos. Aqui, exploramos em profundidade a sintaxe, semântica e exemplos de uso de cada construção da linguagem.

## Declarações

As declarações em ΩMetaLang definem entidades que podem ser referenciadas posteriormente no programa. Elas introduzem nomes no escopo e associam esses nomes a valores, tipos ou comportamentos.

### Declaração de Programa

Todo programa ΩMetaLang começa com uma declaração de programa que define o nome e o escopo global:

```
Omega NomeDoPrograma {
    // Corpo do programa
}
```

**Características:**
- Define o escopo global do programa
- O nome deve ser um identificador válido
- Todas as outras declarações devem estar contidas dentro do corpo do programa

**Exemplo:**
```
Omega ClassificacaoIris {
    // Declarações do programa
}
```

### Declaração de Variáveis

As variáveis em ΩMetaLang são declaradas usando a palavra-chave `var`, seguida pelo nome, tipo opcional e valor inicial:

```
var nome: Tipo = valor;
```

**Características:**
- O tipo pode ser omitido se puder ser inferido do valor inicial
- O valor inicial é obrigatório
- As variáveis são mutáveis por padrão
- O escopo da variável é o bloco em que foi declarada

**Exemplos:**
```
var x: Int = 42;
var y = 3.14;  // Tipo inferido como Float
var nome: String = "Omega";
var ativo = true;  // Tipo inferido como Bool
var numeros: Vector<Int> = [1, 2, 3, 4, 5];
```

### Declaração de Constantes

As constantes são declaradas usando a palavra-chave `const`, seguida pelo nome, tipo opcional e valor:

```
const nome: Tipo = valor;
```

**Características:**
- O valor não pode ser alterado após a declaração
- O tipo pode ser omitido se puder ser inferido do valor
- O valor deve ser conhecido em tempo de compilação
- Constantes podem ser usadas em contextos que exigem valores em tempo de compilação

**Exemplos:**
```
const PI: Float = 3.14159265359;
const MAX_ITERATIONS = 1000;  // Tipo inferido como Int
const MODELO_NOME: String = "RegressaoLogistica";
const DIMENSOES: Vector<Int> = [28, 28];  // Dimensões de uma imagem
```

### Declaração de Modelos

Os modelos são declarados usando a palavra-chave `model`, seguida pelo nome, tipos de entrada/saída e corpo:

```
model NomeDoModelo<TipoEntrada, TipoSaída> {
    // Parâmetros, configurações e métodos
}
```

**Características:**
- Define um modelo de aprendizado de máquina ou processamento de dados
- Especifica os tipos de entrada e saída como parâmetros genéricos
- O corpo pode conter parâmetros, configurações e métodos
- Parâmetros são variáveis treináveis do modelo
- Configurações são hiperparâmetros não treináveis

**Exemplos:**
```
model LogisticRegression<Vector<Float>, Int> {
    // Parâmetros (treináveis)
    param weights: Matrix<Float> = initialize.xavier(input_dim, output_dim);
    param bias: Vector<Float> = initialize.zeros(output_dim);
    
    // Configurações (hiperparâmetros)
    config learning_rate: Float = 0.01;
    config regularization: String = "l2";
    config reg_strength: Float = 0.001;
    
    // Método de forward pass
    function forward(x: Vector<Float>) -> Vector<Float> {
        return softmax(x * weights + bias);
    }
    
    // Método de predição
    function predict(x: Vector<Float>) -> Int {
        var probs = forward(x);
        return argmax(probs);
    }
}
```

### Declaração de Dados

Os dados são declarados usando a palavra-chave `data`, seguida pelo nome e corpo:

```
data NomeDoConjunto {
    // Especificações de fonte, formato, pré-processamento, etc.
}
```

**Características:**
- Define um conjunto de dados para treinamento, validação ou teste
- Especifica a fonte dos dados, formato, características, etc.
- Pode incluir operações de pré-processamento
- Pode definir divisões de dados (train/val/test)

**Exemplos:**
```
data MNISTDataset {
    source: "datasets/mnist.csv";
    format: "csv";
    
    features: {
        columns: [1:784],  // Colunas 1 a 784 são pixels
        shape: [28, 28],   // Reshape para matriz 28x28
        normalize: "min_max"  // Normalização min-max (0-1)
    };
    
    target: {
        column: 0,  // Coluna 0 é o rótulo
        type: "categorical",
        classes: 10  // 10 classes (dígitos 0-9)
    };
    
    split: {
        train: 0.8,
        validation: 0.1,
        test: 0.1,
        seed: 42  // Semente para reprodutibilidade
    };
}
```

### Declaração de Objetivos

Os objetivos são declarados usando a palavra-chave `objective`, seguida pelo nome e corpo:

```
objective NomeDoObjetivo {
    // Função de perda, restrições, métricas, etc.
}
```

**Características:**
- Define o objetivo de otimização para treinamento de modelos
- Especifica a função de perda principal
- Pode incluir termos de regularização
- Pode definir restrições e métricas adicionais

**Exemplos:**
```
objective ClassificationObjective {
    // Função de perda principal
    minimize: cross_entropy(model.forward(data.features), data.target);
    
    // Regularização
    regularize: l2(model.weights, 0.001);
    
    // Restrições
    constraints: [
        model_size < 10MB,  // Restrição de tamanho
        inference_time < 100ms  // Restrição de tempo
    ];
    
    // Métricas para monitoramento
    metrics: {
        accuracy: accuracy(model.predict(data.features), data.target),
        f1: f1_score(model.predict(data.features), data.target),
        precision: precision(model.predict(data.features), data.target),
        recall: recall(model.predict(data.features), data.target)
    };
}
```

### Declaração de Experimentos

Os experimentos são declarados usando a palavra-chave `experiment`, seguida pelo nome e corpo:

```
experiment NomeDoExperimento {
    // Configuração do experimento
}
```

**Características:**
- Define um experimento completo que combina modelo, dados e objetivo
- Especifica configurações de treinamento
- Pode incluir busca de hiperparâmetros
- Define métricas e critérios de avaliação

**Exemplos:**
```
experiment MNISTClassification {
    // Componentes principais
    model: CNNModel;
    data: MNISTDataset;
    objective: ClassificationObjective;
    
    // Configuração de treinamento
    train: {
        optimizer: Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999
        },
        batch_size: 64,
        epochs: 10,
        early_stopping: {
            monitor: "validation_loss",
            patience: 3,
            min_delta: 0.001
        }
    };
    
    // Configuração de avaliação
    evaluate: {
        metrics: ["accuracy", "f1_score", "confusion_matrix"],
        on: "test"  // Avaliar no conjunto de teste
    };
    
    // Configuração de logging
    log: {
        frequency: "epoch",
        metrics: ["train_loss", "validation_loss", "validation_accuracy"],
        save_model: {
            frequency: "best",
            monitor: "validation_accuracy",
            mode: "max"
        }
    };
}
```

### Declaração de Funções

As funções são declaradas usando a palavra-chave `function`, seguida pelo nome, parâmetros, tipo de retorno e corpo:

```
function nome(param1: Tipo1, param2: Tipo2, ...) -> TipoRetorno {
    // Corpo da função
}
```

**Características:**
- Define uma função reutilizável
- Especifica os parâmetros com seus tipos
- Especifica o tipo de retorno
- O corpo contém as instruções a serem executadas

**Exemplos:**
```
function calcularDistancia(p1: Vector<Float>, p2: Vector<Float>) -> Float {
    var soma = 0.0;
    for (var i = 0; i < p1.length(); i++) {
        var diff = p1[i] - p2[i];
        soma = soma + diff * diff;
    }
    return sqrt(soma);
}

function normalizar(vetor: Vector<Float>) -> Vector<Float> {
    var min_val = min(vetor);
    var max_val = max(vetor);
    var range = max_val - min_val;
    
    if (range == 0.0) {
        return vetor.map(_ => 0.0);
    }
    
    return vetor.map(x => (x - min_val) / range);
}
```

### Declaração de Estruturas

As estruturas são declaradas usando a palavra-chave `struct`, seguida pelo nome e corpo:

```
struct NomeDaEstrutura {
    // Campos e métodos
}
```

**Características:**
- Define um tipo de dados personalizado
- Especifica campos com seus tipos
- Pode incluir métodos associados à estrutura
- Instâncias são criadas usando a sintaxe `NomeDaEstrutura{campo1: valor1, ...}`

**Exemplos:**
```
struct Ponto {
    x: Float;
    y: Float;
    
    function distancia(outro: Ponto) -> Float {
        var dx = x - outro.x;
        var dy = y - outro.y;
        return sqrt(dx * dx + dy * dy);
    }
    
    function transladar(dx: Float, dy: Float) -> Ponto {
        return Ponto{x: x + dx, y: y + dy};
    }
}

var p1 = Ponto{x: 1.0, y: 2.0};
var p2 = Ponto{x: 4.0, y: 6.0};
var d = p1.distancia(p2);  // d = 5.0
var p3 = p1.transladar(2.0, 3.0);  // p3 = Ponto{x: 3.0, y: 5.0}
```

### Declaração de Enumerações

As enumerações são declaradas usando a palavra-chave `enum`, seguida pelo nome e variantes:

```
enum NomeDaEnumeracao {
    Variante1,
    Variante2,
    Variante3(Tipo),
    ...
}
```

**Características:**
- Define um tipo que pode ter um valor dentre várias variantes
- As variantes podem ser simples ou conter dados associados
- Instâncias são criadas usando a sintaxe `NomeDaEnumeracao.Variante`
- Útil para representar estados, resultados ou categorias mutuamente exclusivas

**Exemplos:**
```
enum Resultado {
    Sucesso,
    Erro(String)
}

function dividir(a: Float, b: Float) -> Resultado {
    if (b == 0.0) {
        return Resultado.Erro("Divisão por zero");
    } else {
        return Resultado.Sucesso;
    }
}

var res = dividir(10.0, 0.0);
if (res is Resultado.Erro) {
    var Resultado.Erro(mensagem) = res;  // Extração do valor associado
    print("Erro: " + mensagem);
}

enum Forma {
    Circulo(Float),  // Raio
    Retangulo(Float, Float),  // Largura, Altura
    Triangulo(Float, Float, Float)  // Lados a, b, c
}

function calcularArea(forma: Forma) -> Float {
    match forma {
        Forma.Circulo(r) => PI * r * r,
        Forma.Retangulo(w, h) => w * h,
        Forma.Triangulo(a, b, c) => {
            var s = (a + b + c) / 2.0;
            return sqrt(s * (s - a) * (s - b) * (s - c));  // Fórmula de Heron
        }
    }
}
```

## Expressões

As expressões em ΩMetaLang são construções que produzem valores. Elas podem ser combinadas de várias maneiras para formar expressões mais complexas.

### Expressões Literais

Representam valores constantes diretamente no código:

**Inteiros:**
```
42
-7
0
1_000_000  // Underscores para legibilidade
0xFF       // Hexadecimal
0b1010     // Binário
```

**Ponto Flutuante:**
```
3.14159
-0.5
1.0
1e-10      // Notação científica
1.5e3      // 1500.0
```

**Booleanos:**
```
true
false
```

**Strings:**
```
"Texto simples"
"Linha 1\nLinha 2"  // Com caracteres de escape
"""
Texto com múltiplas
linhas sem precisar
de caracteres de escape
"""
```

**Vetores:**
```
[1, 2, 3, 4, 5]
["a", "b", "c"]
[true, false, true]
[]  // Vetor vazio
```

**Matrizes:**
```
[
    [1, 2, 3],
    [4, 5, 6]
]
```

**Mapas:**
```
{"chave1": "valor1", "chave2": "valor2"}
{1: "um", 2: "dois", 3: "três"}
{}  // Mapa vazio
```

**Tuplas:**
```
(1, "texto", true)
(3.14,)  // Tupla com um único elemento
()       // Tupla vazia
```

### Expressões de Acesso

Permitem acessar elementos de estruturas de dados compostas:

**Acesso a Variáveis:**
```
x  // Acessa o valor da variável x
```

**Acesso a Campos:**
```
ponto.x  // Acessa o campo x da estrutura ponto
pessoa.nome  // Acessa o campo nome da estrutura pessoa
```

**Acesso a Elementos:**
```
vetor[0]       // Acessa o primeiro elemento do vetor
matriz[1, 2]   // Acessa o elemento na linha 1, coluna 2
mapa["chave"]  // Acessa o valor associado à chave
tupla.0        // Acessa o primeiro elemento da tupla
```

**Acesso a Métodos:**
```
vetor.length()  // Chama o método length do vetor
ponto.distancia(outro_ponto)  // Chama o método distancia do ponto
```

### Expressões Aritméticas

Realizam operações matemáticas:

**Operações Básicas:**
```
a + b  // Adição
a - b  // Subtração
a * b  // Multiplicação
a / b  // Divisão
a % b  // Módulo (resto da divisão)
a ^ b  // Potência
-a     // Negação
```

**Operações Compostas:**
```
a + b * c          // Multiplicação tem precedência sobre adição
(a + b) * c        // Parênteses alteram a precedência
a + b + c          // Associatividade à esquerda
a * (b + c) / d    // Expressão complexa
```

**Operações com Atribuição:**
```
x += 5   // Equivalente a x = x + 5
x -= 3   // Equivalente a x = x - 3
x *= 2   // Equivalente a x = x * 2
x /= 4   // Equivalente a x = x / 4
x %= 10  // Equivalente a x = x % 10
```

### Expressões Lógicas

Realizam operações booleanas:

**Operações Básicas:**
```
a && b  // AND lógico
a || b  // OR lógico
!a      // NOT lógico
```

**Comparações:**
```
a == b  // Igualdade
a != b  // Desigualdade
a < b   // Menor que
a > b   // Maior que
a <= b  // Menor ou igual a
a >= b  // Maior ou igual a
```

**Operações Compostas:**
```
(a > 0) && (b < 10)           // AND com comparações
(x == 0) || (y == 0) || (z == 0)  // OR com comparações
!(a < b)                       // NOT com comparação
```

### Expressões Condicionais

Permitem execução condicional:

**If-Else:**
```
if (condição) {
    // Código executado se condição for verdadeira
} else {
    // Código executado se condição for falsa
}
```

**If-Else If-Else:**
```
if (condição1) {
    // Código executado se condição1 for verdadeira
} else if (condição2) {
    // Código executado se condição1 for falsa e condição2 for verdadeira
} else {
    // Código executado se ambas as condições forem falsas
}
```

**Expressão Ternária:**
```
condição ? valor_se_verdadeiro : valor_se_falso
```

**Expressão Match:**
```
match valor {
    padrão1 => expressão1,
    padrão2 => expressão2,
    _ => expressão_padrão  // Caso padrão (wildcard)
}
```

**Exemplos de Match:**
```
var resultado = match x {
    0 => "zero",
    1 => "um",
    2 => "dois",
    _ => "outro número"
};

var area = match forma {
    Forma.Circulo(r) => PI * r * r,
    Forma.Retangulo(w, h) => w * h,
    Forma.Triangulo(a, b, c) => {
        var s = (a + b + c) / 2.0;
        return sqrt(s * (s - a) * (s - b) * (s - c));
    }
};
```

### Expressões de Iteração

Permitem repetição de código:

**For Loop:**
```
for (var i = 0; i < 10; i++) {
    // Código repetido 10 vezes
}
```

**For-In Loop:**
```
for (var elemento in vetor) {
    // Código executado para cada elemento do vetor
}

for (var (chave, valor) in mapa) {
    // Código executado para cada par chave-valor do mapa
}
```

**While Loop:**
```
while (condição) {
    // Código repetido enquanto a condição for verdadeira
}
```

**Do-While Loop:**
```
do {
    // Código executado pelo menos uma vez
} while (condição);
```

**Loop com Break e Continue:**
```
for (var i = 0; i < 100; i++) {
    if (i % 2 == 0) {
        continue;  // Pula para a próxima iteração se i for par
    }
    
    if (i > 50) {
        break;  // Sai do loop se i for maior que 50
    }
    
    // Código executado apenas para números ímpares até 50
}
```

### Expressões Funcionais

Permitem programação em estilo funcional:

**Funções Anônimas (Lambdas):**
```
var dobro = x => x * 2;
var soma = (x, y) => x + y;
var fatorial = n => {
    if (n <= 1) return 1;
    return n * fatorial(n - 1);
};
```

**Map, Filter, Reduce:**
```
// Map: aplica uma função a cada elemento
var numeros = [1, 2, 3, 4, 5];
var dobros = numeros.map(x => x * 2);  // [2, 4, 6, 8, 10]

// Filter: seleciona elementos que satisfazem um predicado
var pares = numeros.filter(x => x % 2 == 0);  // [2, 4]

// Reduce: combina elementos usando uma função
var soma = numeros.reduce((acc, x) => acc + x, 0);  // 15
```

**Composição de Funções:**
```
var f = x => x * 2;
var g = x => x + 1;
var h = x => g(f(x));  // h(x) = (x * 2) + 1

var resultado = h(3);  // resultado = 7
```

### Expressões de Chamada de Função

Invocam funções com argumentos:

**Chamada Básica:**
```
calcularDistancia(p1, p2)
sqrt(25)
print("Olá, mundo!")
```

**Chamada com Argumentos Nomeados:**
```
criarPessoa(nome: "João", idade: 30, altura: 1.75)
```

**Chamada de Método:**
```
vetor.append(42)
ponto.transladar(2.0, 3.0)
```

**Chamada Encadeada:**
```
vetor.filter(x => x > 0).map(x => x * 2).sum()
```

### Expressões de Criação de Objetos

Criam instâncias de tipos compostos:

**Criação de Estrutura:**
```
Ponto{x: 1.0, y: 2.0}
Pessoa{nome: "Maria", idade: 25, altura: 1.65}
```

**Criação de Enumeração:**
```
Resultado.Sucesso
Resultado.Erro("Mensagem de erro")
Forma.Circulo(5.0)
```

**Criação de Vetor:**
```
[1, 2, 3, 4, 5]
[x for x in range(10) if x % 2 == 0]  // Compreensão de lista
```

**Criação de Mapa:**
```
{"a": 1, "b": 2, "c": 3}
{x: x*x for x in range(5)}  // Compreensão de mapa
```

### Expressões de Tratamento de Erros

Lidam com situações excepcionais:

**Try-Catch:**
```
try {
    var resultado = operacaoArriscada();
    // Código que pode lançar erro
} catch (erro) {
    // Código executado se ocorrer um erro
} finally {
    // Código sempre executado, independentemente de erro
}
```

**Propagação de Erro:**
```
function operacaoSegura() -> Resultado {
    var arquivo = abrirArquivo("dados.txt");
    if (arquivo is Erro) {
        return Resultado.Erro("Falha ao abrir arquivo");
    }
    
    var dados = lerDados(arquivo);
    if (dados is Erro) {
        return Resultado.Erro("Falha ao ler dados");
    }
    
    return Resultado.Sucesso;
}
```

**Unwrap de Option:**
```
var valor_opcional: Option<Int> = Some(42);
var valor = valor_opcional.unwrap();  // 42
var valor_seguro = valor_opcional.unwrapOr(0);  // 42

var nenhum_valor: Option<Int> = None;
// nenhum_valor.unwrap();  // Erro em tempo de execução
var valor_padrao = nenhum_valor.unwrapOr(0);  // 0
```

## Precedência e Associatividade

A precedência e associatividade dos operadores em ΩMetaLang seguem as convenções matemáticas padrão:

1. Parênteses `()`
2. Chamadas de função, acesso a membros `.`, acesso a elementos `[]`
3. Operadores unários (`-`, `!`)
4. Potência `^`
5. Multiplicação `*`, divisão `/`, módulo `%`
6. Adição `+`, subtração `-`
7. Comparações `<`, `>`, `<=`, `>=`
8. Igualdade `==`, desigualdade `!=`
9. AND lógico `&&`
10. OR lógico `||`
11. Expressão condicional (ternária) `? :`
12. Atribuições `=`, `+=`, `-=`, etc.

A maioria dos operadores é associativa à esquerda, exceto:
- Potência `^` é associativa à direita
- Atribuições são associativas à direita

## Escopo e Visibilidade

ΩMetaLang segue regras de escopo léxico:

1. **Escopo Global**: Declarações no nível superior do programa
2. **Escopo de Bloco**: Declarações dentro de blocos `{}`
3. **Escopo de Função**: Parâmetros e variáveis locais de funções
4. **Escopo de Estrutura**: Campos e métodos de estruturas

Regras de visibilidade:
- Identificadores declarados em um escopo são visíveis nesse escopo e em todos os escopos aninhados
- Identificadores com o mesmo nome em um escopo interno ocultam (shadow) os do escopo externo
- Declarações de estruturas, enumerações e funções são visíveis em todo o programa

## Exemplos Completos de Expressões e Declarações

### Exemplo 1: Processamento de Dados

```
Omega ProcessamentoDados {
    // Declaração de estrutura para representar um ponto de dados
    struct PontoDados {
        timestamp: Int;
        valor: Float;
        valido: Bool;
        
        function normalizado(min_val: Float, max_val: Float) -> Float {
            if (!valido || max_val == min_val) {
                return 0.0;
            }
            return (valor - min_val) / (max_val - min_val);
        }
    }
    
    // Declaração de função para processar uma série temporal
    function processarSerie(pontos: Vector<PontoDados>) -> Map<String, Float> {
        if (pontos.length() == 0) {
            return {};
        }
        
        // Filtra pontos válidos
        var validos = pontos.filter(p => p.valido);
        if (validos.length() == 0) {
            return {
                "count": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0
            };
        }
        
        // Extrai valores
        var valores = validos.map(p => p.valor);
        
        // Calcula estatísticas
        var min_val = min(valores);
        var max_val = max(valores);
        var mean_val = mean(valores);
        var std_val = std(valores);
        
        // Normaliza valores
        var normalizados = validos.map(p => p.normalizado(min_val, max_val));
        
        // Retorna resultados
        return {
            "count": validos.length(),
            "mean": mean_val,
            "min": min_val,
            "max": max_val,
            "std": std_val,
            "normalized_mean": mean(normalizados)
        };
    }
    
    // Declaração de dados
    data SensorData {
        source: "sensors/temperature.csv";
        columns: {
            "timestamp": Int,
            "temperature": Float,
            "valid": Bool
        };
    }
    
    // Declaração de experimento
    experiment AnaliseSensores {
        // Carrega dados
        var dados_brutos = load(SensorData);
        
        // Converte para pontos de dados
        var pontos = dados_brutos.map(row => 
            PontoDados{
                timestamp: row["timestamp"],
                valor: row["temperature"],
                valido: row["valid"]
            }
        );
        
        // Processa dados
        var resultados = processarSerie(pontos);
        
        // Exibe resultados
        print("Análise de Temperatura:");
        for (var (chave, valor) in resultados) {
            print(f"{chave}: {valor}");
        }
        
        // Salva resultados
        save(resultados, "resultados/temperatura_stats.json");
    }
}
```

### Exemplo 2: Algoritmo de Otimização

```
Omega AlgoritmoGenetico {
    // Declaração de tipos
    type Individuo = Vector<Float>;
    type Populacao = Vector<Individuo>;
    
    // Declaração de constantes
    const TAMANHO_POPULACAO: Int = 100;
    const DIMENSOES: Int = 10;
    const MAX_GERACOES: Int = 500;
    const TAXA_MUTACAO: Float = 0.01;
    const TAXA_CROSSOVER: Float = 0.7;
    
    // Declaração de função de fitness
    function fitness(individuo: Individuo) -> Float {
        // Função de Rosenbrock (banana function)
        var soma = 0.0;
        for (var i = 0; i < individuo.length() - 1; i++) {
            var x = individuo[i];
            var y = individuo[i + 1];
            soma += 100 * (y - x*x)*(y - x*x) + (1 - x)*(1 - x);
        }
        return -soma;  // Negativo porque queremos maximizar
    }
    
    // Declaração de função para inicializar população
    function inicializarPopulacao() -> Populacao {
        var populacao: Populacao = [];
        for (var i = 0; i < TAMANHO_POPULACAO; i++) {
            var individuo: Individuo = [];
            for (var j = 0; j < DIMENSOES; j++) {
                individuo.append(random.uniform(-5.0, 5.0));
            }
            populacao.append(individuo);
        }
        return populacao;
    }
    
    // Declaração de função para seleção
    function selecionar(populacao: Populacao, fitness_valores: Vector<Float>) -> Tuple<Individuo, Individuo> {
        // Seleção por torneio
        var idx1 = random.integer(0, populacao.length() - 1);
        var idx2 = random.integer(0, populacao.length() - 1);
        var pai1 = populacao[idx1];
        
        idx1 = random.integer(0, populacao.length() - 1);
        idx2 = random.integer(0, populacao.length() - 1);
        var pai2 = populacao[idx2];
        
        return (pai1, pai2);
    }
    
    // Declaração de função para crossover
    function crossover(pai1: Individuo, pai2: Individuo) -> Tuple<Individuo, Individuo> {
        if (random.uniform(0.0, 1.0) > TAXA_CROSSOVER) {
            return (pai1, pai2);
        }
        
        var ponto = random.integer(1, DIMENSOES - 1);
        var filho1: Individuo = pai1[0:ponto] + pai2[ponto:];
        var filho2: Individuo = pai2[0:ponto] + pai1[ponto:];
        
        return (filho1, filho2);
    }
    
    // Declaração de função para mutação
    function mutar(individuo: Individuo) -> Individuo {
        var resultado = individuo.copy();
        for (var i = 0; i < resultado.length(); i++) {
            if (random.uniform(0.0, 1.0) < TAXA_MUTACAO) {
                resultado[i] += random.normal(0.0, 0.5);
            }
        }
        return resultado;
    }
    
    // Declaração de experimento
    experiment OtimizacaoRosenbrock {
        // Inicializa população
        var populacao = inicializarPopulacao();
        var melhor_individuo: Individuo = [];
        var melhor_fitness = -Infinity;
        
        // Loop principal
        for (var geracao = 0; geracao < MAX_GERACOES; geracao++) {
            // Avalia fitness
            var fitness_valores = populacao.map(fitness);
            
            // Encontra o melhor
            var idx_melhor = argmax(fitness_valores);
            if (fitness_valores[idx_melhor] > melhor_fitness) {
                melhor_fitness = fitness_valores[idx_melhor];
                melhor_individuo = populacao[idx_melhor];
                print(f"Geração {geracao}: Melhor fitness = {melhor_fitness}");
            }
            
            // Cria nova população
            var nova_populacao: Populacao = [];
            
            // Elitismo: mantém o melhor
            nova_populacao.append(populacao[idx_melhor]);
            
            // Gera novos indivíduos
            while (nova_populacao.length() < TAMANHO_POPULACAO) {
                var (pai1, pai2) = selecionar(populacao, fitness_valores);
                var (filho1, filho2) = crossover(pai1, pai2);
                
                filho1 = mutar(filho1);
                filho2 = mutar(filho2);
                
                nova_populacao.append(filho1);
                if (nova_populacao.length() < TAMANHO_POPULACAO) {
                    nova_populacao.append(filho2);
                }
            }
            
            populacao = nova_populacao;
        }
        
        // Exibe resultado final
        print("Otimização concluída!");
        print(f"Melhor solução: {melhor_individuo}");
        print(f"Fitness: {melhor_fitness}");
    }
}
```

## Boas Práticas

1. **Nomeação**:
   - Use `camelCase` para variáveis, funções e métodos
   - Use `PascalCase` para tipos, estruturas e enumerações
   - Use `UPPER_CASE` para constantes

2. **Formatação**:
   - Use indentação consistente (2 ou 4 espaços)
   - Coloque chaves em novas linhas para blocos grandes
   - Mantenha linhas com comprimento razoável (< 80-100 caracteres)

3. **Comentários**:
   - Documente o propósito de funções, estruturas e blocos complexos
   - Use comentários para explicar algoritmos não triviais
   - Evite comentários óbvios que apenas repetem o código

4. **Organização**:
   - Agrupe declarações relacionadas
   - Coloque tipos e funções auxiliares antes de seu uso
   - Organize o código do mais geral para o mais específico

5. **Estilo Funcional**:
   - Prefira operações funcionais (map, filter, reduce) quando apropriado
   - Minimize efeitos colaterais e mutação de estado
   - Use composição de funções para operações complexas

---

*Este documento é parte da documentação oficial do Projeto Omega e complementa o Guia de Referência ΩMetaLang e a Especificação de Tipos.*
