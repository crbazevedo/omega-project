# Briefings de Exemplo para Testar o ΩScribe

Este diretório contém exemplos de briefings em linguagem natural para testar o agente ΩScribe (US-1.2).
Cada briefing descreve um problema que deve ser convertido em código ΩMetaLang.

## Briefing 1: Classificador Simples
```
Criar um classificador simples para dados MNIST. O modelo deve usar regressão logística
para classificar dígitos manuscritos em 10 classes (0-9). Os dados de entrada são imagens
de 28x28 pixels em escala de cinza, representadas como vetores de 784 dimensões.
```

## Briefing 2: Regressão Linear
```
Implementar um modelo de regressão linear para prever preços de casas. O conjunto de dados
contém 5 características: área, número de quartos, número de banheiros, idade da casa e
distância do centro da cidade. O objetivo é minimizar o erro quadrático médio.
```

## Briefing 3: Previsão de Séries Temporais
```
Desenvolver um modelo para prever valores futuros de uma série temporal de vendas mensais.
Os dados históricos abrangem 5 anos (60 pontos). O modelo deve prever os próximos 12 meses
e minimizar o erro absoluto médio.
```

## Briefing 4: Agente de Reinforcement Learning
```
Criar um agente de reinforcement learning para o ambiente CartPole. O agente deve aprender
a equilibrar um poste em um carrinho, recebendo recompensa por cada passo que o poste
permanece em pé. O estado é representado por 4 variáveis contínuas e há 2 ações possíveis.
```

## Briefing 5: Sistema de Recomendação
```
Implementar um sistema de recomendação baseado em filtragem colaborativa para um serviço
de streaming de música. O sistema deve recomendar músicas para usuários com base em suas
preferências passadas e nas preferências de usuários similares.
```

## Briefing 6: Análise de Sentimento
```
Desenvolver um modelo para análise de sentimento em comentários de produtos. O modelo deve
classificar os comentários em três categorias: negativo, neutro e positivo. Os comentários
são textos em português com comprimento variável.
```

## Briefing 7: Classificação de Imagens
```
Criar um modelo de classificação de imagens para identificar diferentes espécies de plantas
em fotografias. O conjunto de dados contém imagens coloridas de 224x224 pixels e 50 classes
diferentes de plantas.
```

## Briefing 8: Detecção de Anomalias
```
Implementar um sistema de detecção de anomalias para identificar transações fraudulentas.
O sistema deve aprender o padrão normal de transações e sinalizar aquelas que desviam
significativamente desse padrão.
```

## Briefing 9: Clustering
```
Desenvolver um modelo de clustering para segmentar clientes de um e-commerce com base em
seu comportamento de compra. As características incluem frequência de compra, valor médio
gasto, categorias de produtos preferidas e tempo desde a última compra.
```

## Briefing 10: Sistema Híbrido
```
Criar um sistema híbrido que combine classificação de imagens e reinforcement learning para
um robô de seleção de objetos. O sistema deve primeiro identificar objetos em uma imagem e
depois decidir a melhor sequência de ações para pegá-los.
```
