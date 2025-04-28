"""
Módulo para o componente ΩTuner.

O ΩTuner é responsável por otimizar os hiperparâmetros de modelos e experimentos
no sistema Omega. Ele implementa diversos algoritmos de otimização, incluindo:
- Busca em grade (Grid Search)
- Busca aleatória (Random Search)
- Otimização Bayesiana
- Algoritmos evolutivos

Este módulo segue os princípios invariantes do Projeto Omega e integra-se
com outros componentes como ΩArchitect e ΩSynthesizer.
"""

import logging
import random
import numpy as np
from typing import Dict, List, Any, Callable, Tuple, Optional, Union
from enum import Enum

# Configuração de logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


class OptimizationStrategy(Enum):
    """Estratégias de otimização suportadas pelo ΩTuner."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"


class SearchSpace:
    """
    Representa o espaço de busca para otimização de hiperparâmetros.
    
    Atributos:
        parameters (Dict): Dicionário de parâmetros e seus possíveis valores.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Inicializa o espaço de busca.
        
        Args:
            parameters: Dicionário onde as chaves são nomes de parâmetros e os valores
                        podem ser listas de valores discretos ou tuplas (min, max, [step])
                        para valores contínuos.
        """
        self.parameters = parameters
        self._validate_parameters()
        logger.info(f"Espaço de busca inicializado com {len(parameters)} parâmetros")
    
    def _validate_parameters(self):
        """Valida os parâmetros do espaço de busca."""
        for param_name, param_values in self.parameters.items():
            if not isinstance(param_name, str):
                raise ValueError(f"Nome do parâmetro deve ser string, recebido: {type(param_name)}")
            
            # Verifica se os valores são válidos (lista, tupla ou range)
            if not (isinstance(param_values, (list, tuple, range)) or 
                   (isinstance(param_values, dict) and 'min' in param_values and 'max' in param_values)):
                raise ValueError(f"Valores para {param_name} devem ser lista, tupla, range ou dict com min/max")
    
    def sample_random(self) -> Dict[str, Any]:
        """
        Amostra aleatoriamente um ponto no espaço de busca.
        
        Returns:
            Dict[str, Any]: Dicionário com valores amostrados para cada parâmetro.
        """
        sampled_params = {}
        
        for param_name, param_values in self.parameters.items():
            if isinstance(param_values, list):
                # Amostra de lista discreta
                sampled_params[param_name] = random.choice(param_values)
            elif isinstance(param_values, tuple) and len(param_values) >= 2:
                # Amostra de intervalo contínuo
                min_val, max_val = param_values[0], param_values[1]
                log_scale = False
                
                # Verifica se há um terceiro elemento indicando escala logarítmica
                if len(param_values) > 2 and isinstance(param_values[2], dict):
                    log_scale = param_values[2].get('log', False)
                
                if log_scale:
                    # Amostragem em escala logarítmica
                    if min_val <= 0:
                        raise ValueError(f"Valores mínimos para escala logarítmica devem ser positivos: {param_name}")
                    log_min, log_max = np.log(min_val), np.log(max_val)
                    sampled_params[param_name] = float(np.exp(random.uniform(log_min, log_max)))
                else:
                    # Amostragem em escala linear
                    sampled_params[param_name] = float(random.uniform(min_val, max_val))
            elif isinstance(param_values, dict) and 'min' in param_values and 'max' in param_values:
                # Formato alternativo para intervalos
                min_val, max_val = param_values['min'], param_values['max']
                log_scale = param_values.get('log', False)
                
                if log_scale:
                    if min_val <= 0:
                        raise ValueError(f"Valores mínimos para escala logarítmica devem ser positivos: {param_name}")
                    log_min, log_max = np.log(min_val), np.log(max_val)
                    sampled_params[param_name] = float(np.exp(random.uniform(log_min, log_max)))
                else:
                    sampled_params[param_name] = float(random.uniform(min_val, max_val))
            elif isinstance(param_values, range):
                # Amostra de range
                sampled_params[param_name] = random.choice(list(param_values))
        
        return sampled_params
    
    def get_grid_points(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Gera pontos em grade no espaço de busca.
        
        Args:
            num_samples: Número máximo de pontos a serem gerados (opcional).
                         Se None, gera todos os pontos possíveis.
        
        Returns:
            List[Dict[str, Any]]: Lista de dicionários com combinações de parâmetros.
        """
        # Converte todos os parâmetros para listas discretas
        discrete_params = {}
        
        for param_name, param_values in self.parameters.items():
            if isinstance(param_values, list):
                discrete_params[param_name] = param_values
            elif isinstance(param_values, tuple) and len(param_values) >= 2:
                min_val, max_val = param_values[0], param_values[1]
                
                # Verifica se há um terceiro elemento com informações adicionais
                num_points = 10  # Valor padrão
                log_scale = False
                
                if len(param_values) > 2 and isinstance(param_values[2], dict):
                    num_points = param_values[2].get('num_points', 10)
                    log_scale = param_values[2].get('log', False)
                
                if log_scale:
                    if min_val <= 0:
                        raise ValueError(f"Valores mínimos para escala logarítmica devem ser positivos: {param_name}")
                    log_min, log_max = np.log(min_val), np.log(max_val)
                    points = np.exp(np.linspace(log_min, log_max, num_points))
                else:
                    points = np.linspace(min_val, max_val, num_points)
                
                discrete_params[param_name] = points.tolist()
            elif isinstance(param_values, dict) and 'min' in param_values and 'max' in param_values:
                min_val, max_val = param_values['min'], param_values['max']
                num_points = param_values.get('num_points', 10)
                log_scale = param_values.get('log', False)
                
                if log_scale:
                    if min_val <= 0:
                        raise ValueError(f"Valores mínimos para escala logarítmica devem ser positivos: {param_name}")
                    log_min, log_max = np.log(min_val), np.log(max_val)
                    points = np.exp(np.linspace(log_min, log_max, num_points))
                else:
                    points = np.linspace(min_val, max_val, num_points)
                
                discrete_params[param_name] = points.tolist()
            elif isinstance(param_values, range):
                discrete_params[param_name] = list(param_values)
        
        # Calcula o produto cartesiano
        import itertools
        param_names = list(discrete_params.keys())
        param_values = [discrete_params[name] for name in param_names]
        
        # Calcula o número total de combinações
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        logger.info(f"Gerando grade com {total_combinations} combinações de parâmetros")
        
        # Se num_samples for especificado e menor que o total, amostra aleatoriamente
        if num_samples is not None and num_samples < total_combinations:
            logger.info(f"Limitando a {num_samples} amostras aleatórias da grade")
            all_combinations = list(itertools.product(*param_values))
            sampled_indices = random.sample(range(len(all_combinations)), num_samples)
            combinations = [all_combinations[i] for i in sampled_indices]
        else:
            combinations = itertools.product(*param_values)
        
        # Converte para lista de dicionários
        result = []
        for combination in combinations:
            param_dict = {name: value for name, value in zip(param_names, combination)}
            result.append(param_dict)
        
        return result


class OmegaTuner:
    """
    Implementação principal do componente ΩTuner.
    
    O ΩTuner é responsável por otimizar hiperparâmetros de modelos e experimentos
    no sistema Omega, buscando maximizar métricas de desempenho.
    """
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.RANDOM_SEARCH):
        """
        Inicializa o ΩTuner com a estratégia de otimização especificada.
        
        Args:
            strategy: Estratégia de otimização a ser utilizada.
        """
        self.strategy = strategy
        self.best_params = None
        self.best_score = float('-inf')
        self.results_history = []
        logger.info(f"ΩTuner inicializado com estratégia: {strategy.value}")
    
    def optimize(self, 
                objective_function: Callable[[Dict[str, Any]], float],
                search_space: SearchSpace,
                max_iterations: int = 10,
                maximize: bool = True,
                early_stopping: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executa a otimização de hiperparâmetros.
        
        Args:
            objective_function: Função que recebe um conjunto de parâmetros e retorna uma métrica.
            search_space: Espaço de busca para os hiperparâmetros.
            max_iterations: Número máximo de iterações/avaliações.
            maximize: Se True, maximiza a função objetivo; se False, minimiza.
            early_stopping: Configuração para parada antecipada (opcional).
        
        Returns:
            Dict[str, Any]: Melhores parâmetros encontrados.
        """
        logger.info(f"Iniciando otimização com {max_iterations} iterações máximas")
        
        # Reinicia o histórico e melhor resultado
        self.results_history = []
        self.best_score = float('-inf') if maximize else float('inf')
        self.best_params = None
        
        # Configura multiplicador para maximização/minimização
        score_multiplier = 1 if maximize else -1
        
        # Implementa diferentes estratégias de otimização
        if self.strategy == OptimizationStrategy.GRID_SEARCH:
            return self._optimize_grid_search(objective_function, search_space, 
                                             max_iterations, score_multiplier, early_stopping)
        elif self.strategy == OptimizationStrategy.RANDOM_SEARCH:
            return self._optimize_random_search(objective_function, search_space, 
                                               max_iterations, score_multiplier, early_stopping)
        elif self.strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
            logger.warning("Otimização Bayesiana ainda não implementada completamente. Usando busca aleatória.")
            return self._optimize_random_search(objective_function, search_space, 
                                               max_iterations, score_multiplier, early_stopping)
        elif self.strategy == OptimizationStrategy.EVOLUTIONARY:
            logger.warning("Algoritmos evolutivos ainda não implementados completamente. Usando busca aleatória.")
            return self._optimize_random_search(objective_function, search_space, 
                                               max_iterations, score_multiplier, early_stopping)
        else:
            raise ValueError(f"Estratégia de otimização não suportada: {self.strategy}")
    
    def _optimize_grid_search(self,
                             objective_function: Callable[[Dict[str, Any]], float],
                             search_space: SearchSpace,
                             max_iterations: int,
                             score_multiplier: int,
                             early_stopping: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Implementa a estratégia de busca em grade.
        
        Args:
            objective_function: Função objetivo a ser otimizada.
            search_space: Espaço de busca para os hiperparâmetros.
            max_iterations: Número máximo de iterações.
            score_multiplier: 1 para maximização, -1 para minimização.
            early_stopping: Configuração para parada antecipada.
            
        Returns:
            Dict[str, Any]: Melhores parâmetros encontrados.
        """
        logger.info("Executando otimização com busca em grade")
        
        # Gera pontos da grade, limitados pelo número máximo de iterações
        grid_points = search_space.get_grid_points(max_iterations)
        
        # Configura contadores para parada antecipada
        no_improvement_count = 0
        patience = early_stopping.get('patience', float('inf')) if early_stopping else float('inf')
        min_delta = early_stopping.get('min_delta', 0) if early_stopping else 0
        
        # Avalia cada ponto da grade
        for i, params in enumerate(grid_points):
            # Avalia a função objetivo
            score = objective_function(params)
            adjusted_score = score * score_multiplier
            
            # Registra resultado
            self.results_history.append({
                'iteration': i,
                'params': params,
                'score': score
            })
            
            # Atualiza melhor resultado
            if self.best_params is None or adjusted_score > self.best_score * score_multiplier:
                improvement = adjusted_score - self.best_score * score_multiplier if self.best_params else float('inf')
                
                if improvement > min_delta:
                    logger.info(f"Nova melhor pontuação: {score} com parâmetros: {params}")
                    self.best_score = score
                    self.best_params = params.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # Verifica critério de parada antecipada
            if no_improvement_count >= patience:
                logger.info(f"Parando antecipadamente após {i+1} iterações sem melhoria significativa")
                break
        
        logger.info(f"Otimização concluída. Melhor pontuação: {self.best_score}")
        return self.best_params
    
    def _optimize_random_search(self,
                               objective_function: Callable[[Dict[str, Any]], float],
                               search_space: SearchSpace,
                               max_iterations: int,
                               score_multiplier: int,
                               early_stopping: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Implementa a estratégia de busca aleatória.
        
        Args:
            objective_function: Função objetivo a ser otimizada.
            search_space: Espaço de busca para os hiperparâmetros.
            max_iterations: Número máximo de iterações.
            score_multiplier: 1 para maximização, -1 para minimização.
            early_stopping: Configuração para parada antecipada.
            
        Returns:
            Dict[str, Any]: Melhores parâmetros encontrados.
        """
        logger.info("Executando otimização com busca aleatória")
        
        # Configura contadores para parada antecipada
        no_improvement_count = 0
        patience = early_stopping.get('patience', float('inf')) if early_stopping else float('inf')
        min_delta = early_stopping.get('min_delta', 0) if early_stopping else 0
        
        # Executa iterações de busca aleatória
        for i in range(max_iterations):
            # Amostra aleatoriamente do espaço de busca
            params = search_space.sample_random()
            
            # Avalia a função objetivo
            score = objective_function(params)
            adjusted_score = score * score_multiplier
            
            # Registra resultado
            self.results_history.append({
                'iteration': i,
                'params': params,
                'score': score
            })
            
            # Atualiza melhor resultado
            if self.best_params is None or adjusted_score > self.best_score * score_multiplier:
                improvement = adjusted_score - self.best_score * score_multiplier if self.best_params else float('inf')
                
                if improvement > min_delta:
                    logger.info(f"Nova melhor pontuação: {score} com parâmetros: {params}")
                    self.best_score = score
                    self.best_params = params.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # Verifica critério de parada antecipada
            if no_improvement_count >= patience:
                logger.info(f"Parando antecipadamente após {i+1} iterações sem melhoria significativa")
                break
        
        logger.info(f"Otimização concluída. Melhor pontuação: {self.best_score}")
        return self.best_params
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo dos resultados da otimização.
        
        Returns:
            Dict[str, Any]: Resumo dos resultados, incluindo melhores parâmetros,
                           melhor pontuação e estatísticas do histórico.
        """
        if not self.results_history:
            return {"status": "Nenhuma otimização realizada ainda"}
        
        # Extrai pontuações do histórico
        scores = [result['score'] for result in self.results_history]
        
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "num_iterations": len(self.results_history),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "strategy": self.strategy.value
        }


# Exemplo de uso
if __name__ == "__main__":
    # Configura logging
    logging.basicConfig(level=logging.INFO)
    
    # Define uma função objetivo simples para teste
    def objective_function(params):
        """Função objetivo de exemplo: uma parábola com ruído."""
        x = params['x']
        y = params['y']
        z = -(x - 3)**2 - (y - 4)**2 + random.uniform(-0.1, 0.1)
        return z
    
    # Define o espaço de busca
    search_space = SearchSpace({
        'x': (0, 10),  # Intervalo contínuo
        'y': [1, 2, 3, 4, 5, 6, 7]  # Valores discretos
    })
    
    # Cria e executa o tuner
    tuner = OmegaTuner(strategy=OptimizationStrategy.RANDOM_SEARCH)
    best_params = tuner.optimize(
        objective_function=objective_function,
        search_space=search_space,
        max_iterations=50,
        maximize=True,
        early_stopping={'patience': 10, 'min_delta': 0.01}
    )
    
    # Exibe resultados
    print("\n=== Resultados da Otimização ===")
    print(f"Melhores parâmetros: {best_params}")
    print(f"Melhor pontuação: {tuner.best_score}")
    
    # Exibe resumo
    summary = tuner.get_results_summary()
    print("\n=== Resumo da Otimização ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
