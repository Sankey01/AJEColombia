"""
Data Loader Module

Módulo simple para cargar archivos CSV y convertirlos a DataFrame.
"""

import pandas as pd


def load_payment_data(filepath: str = r'C:\Users\Kenny\PycharmProjects\AJEColombia\resource\condiciones_pagos (1).csv') -> pd.DataFrame:
    """
    Carga el archivo CSV de condiciones de pagos.

    Args:
        filepath: Ruta al archivo CSV (default: ruta del proyecto)

    Returns:
        DataFrame con los datos cargados
    """
    df = pd.read_csv(filepath)
    return df