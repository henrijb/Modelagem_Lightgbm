"""
Módulo de coleta de dados
"""

from .sidra_1612 import carregar_tabela_1612, coletar_soja_sidra_rs
from .noaa import coletar_noaa_oni
from .inmet_estacoes import processar_todas_estacoes

__all__ = [
    'carregar_tabela_1612',
    'coletar_soja_sidra_rs',
    'coletar_noaa_oni',
    'processar_todas_estacoes'
]