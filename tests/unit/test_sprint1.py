import unittest
import sys
import os
import logging

# Configurar o path para importar os módulos do projeto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.languages.metalang_parser import OmegaMetaLangParser
from src.agents.cognitive_agents import OmegaScribe

# Desativar logs durante os testes
logging.disable(logging.CRITICAL)

class TestOmegaScribe(unittest.TestCase):
    """Testes unitários para o agente ΩScribe."""
    
    def setUp(self):
        """Configuração inicial para cada teste."""
        self.scribe = OmegaScribe()
        self.parser = OmegaMetaLangParser()
    
    def test_classification_briefing(self):
        """Testa a conversão de um briefing de classificação."""
        briefing = "Criar um classificador simples para dados MNIST."
        metalang_code = self.scribe.process(briefing)
        
        # Verificar se o código contém elementos esperados
        self.assertIn("OmegaSpec", metalang_code)
        self.assertIn("GeneratedClassifier", metalang_code)
        self.assertIn("LogisticRegression", metalang_code)
        
        # Verificar se o código é parsável
        try:
            ast = self.parser.parse(metalang_code)
            self.assertIsNotNone(ast)
            self.assertEqual(ast["type"], "OmegaProgram")
        except Exception as e:
            self.fail(f"Parser falhou ao analisar o código gerado: {e}")
    
    def test_regression_briefing(self):
        """Testa a conversão de um briefing de regressão."""
        briefing = "Implementar um modelo de regressão linear."
        metalang_code = self.scribe.process(briefing)
        
        # Verificar se o código contém elementos esperados
        self.assertIn("OmegaSpec", metalang_code)
        self.assertIn("GeneratedRegressor", metalang_code)
        self.assertIn("LinearRegression", metalang_code)
        
        # Verificar se o código é parsável
        try:
            ast = self.parser.parse(metalang_code)
            self.assertIsNotNone(ast)
            self.assertEqual(ast["type"], "OmegaProgram")
        except Exception as e:
            self.fail(f"Parser falhou ao analisar o código gerado: {e}")
    
    def test_generic_briefing(self):
        """Testa a conversão de um briefing genérico."""
        briefing = "Desenvolver um sistema genérico."
        metalang_code = self.scribe.process(briefing)
        
        # Verificar se o código contém elementos esperados
        self.assertIn("OmegaSpec", metalang_code)
        self.assertIn("GeneratedSpec", metalang_code)
        
        # Verificar se o código é parsável
        try:
            ast = self.parser.parse(metalang_code)
            self.assertIsNotNone(ast)
            self.assertEqual(ast["type"], "OmegaProgram")
        except Exception as e:
            self.fail(f"Parser falhou ao analisar o código gerado: {e}")


class TestOmegaMetaLangParser(unittest.TestCase):
    """Testes unitários para o parser ΩMetaLang."""
    
    def setUp(self):
        """Configuração inicial para cada teste."""
        self.parser = OmegaMetaLangParser()
    
    def test_simple_program(self):
        """Testa a análise de um programa ΩMetaLang simples."""
        code = """
        OmegaSpec SimpleTest {
            Variable x : Int = 10;
            
            Model dummy {
                Input: x;
                Output: x;
                Structure: DefaultModel;
                Loss: DefaultLoss(x);
            }
            
            Objective {
                Minimize: S;
                WeightS: 1.0;
                WeightA: 0.0;
                WeightE: 0.1;
            }
        }
        """
        
        try:
            ast = self.parser.parse(code)
            self.assertIsNotNone(ast)
            self.assertEqual(ast["type"], "OmegaProgram")
            self.assertEqual(ast["header"]["name"], "SimpleTest")
        except Exception as e:
            self.fail(f"Parser falhou ao analisar código simples: {e}")
    
    def test_variable_declaration(self):
        """Testa a análise de declarações de variáveis."""
        code = """
        OmegaSpec VarTest {
            Variable x : Int = 10;
            Variable y : Float;
            Variable z : Matrix<Float, 3, 4>;
            
            Objective {
                Minimize: S;
            }
        }
        """
        
        try:
            ast = self.parser.parse(code)
            self.assertIsNotNone(ast)
            
            # Verificar se as declarações de variáveis foram analisadas corretamente
            declarations = ast["declarations"]["items"]
            self.assertEqual(len(declarations), 3)
            
            # Verificar a primeira variável (com valor inicial)
            self.assertEqual(declarations[0]["type"], "VariableDeclaration")
            self.assertEqual(declarations[0]["name"], "x")
            self.assertEqual(declarations[0]["var_type"]["base_type"], "Int")
            self.assertIsNotNone(declarations[0]["initial_value"])
            
            # Verificar a segunda variável (sem valor inicial)
            self.assertEqual(declarations[1]["type"], "VariableDeclaration")
            self.assertEqual(declarations[1]["name"], "y")
            self.assertEqual(declarations[1]["var_type"]["base_type"], "Float")
            self.assertIsNone(declarations[1]["initial_value"])
            
            # Verificar a terceira variável (tipo complexo)
            self.assertEqual(declarations[2]["type"], "VariableDeclaration")
            self.assertEqual(declarations[2]["name"], "z")
            self.assertEqual(declarations[2]["var_type"]["base_type"], "Matrix")
            
        except Exception as e:
            self.fail(f"Parser falhou ao analisar declarações de variáveis: {e}")


class TestIntegration(unittest.TestCase):
    """Testes de integração para o fluxo Scribe -> Parser."""
    
    def setUp(self):
        """Configuração inicial para cada teste."""
        self.scribe = OmegaScribe()
        self.parser = OmegaMetaLangParser()
    
    def test_scribe_parser_integration(self):
        """Testa o fluxo completo de Scribe -> Parser com diferentes briefings."""
        briefings = [
            "Criar um classificador simples para dados MNIST.",
            "Implementar um modelo de regressão linear.",
            "Desenvolver um sistema genérico."
        ]
        
        for briefing in briefings:
            with self.subTest(briefing=briefing):
                # 1. Scribe gera código ΩMetaLang
                metalang_code = self.scribe.process(briefing)
                
                # 2. Parser analisa o código gerado
                try:
                    ast = self.parser.parse(metalang_code)
                    self.assertIsNotNone(ast)
                    self.assertEqual(ast["type"], "OmegaProgram")
                except Exception as e:
                    self.fail(f"Falha na integração Scribe -> Parser para briefing '{briefing}': {e}")


if __name__ == '__main__':
    unittest.main()
