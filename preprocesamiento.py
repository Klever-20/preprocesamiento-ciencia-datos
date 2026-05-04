import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def cargar_datos(ruta_archivo):
    """Carga un dataset desde un archivo CSV."""
    return pd.read_csv(ruta_archivo)

def manejar_nulos(df, estrategia='media'):
    """Maneja valores nulos en el DataFrame."""
    if estrategia == 'eliminar':
        return df.dropna()
    elif estrategia == 'media':
        return df.fillna(df.mean(numeric_only=True))
    elif estrategia == 'mediana':
        return df.fillna(df.median(numeric_only=True))
    else:
        raise ValueError("Estrategia no valida")

def normalizar_datos(df, columnas):
    """Aplica normalizacion estandar a columnas numericas."""
    scaler = StandardScaler()
    df[columnas] = scaler.fit_transform(df[columnas])
    return df

def codificar_categoricas(df, columnas):
    """Codifica variables categoricas usando LabelEncoder."""
    le = LabelEncoder()
    for col in columnas:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def eliminar_duplicados(df):
    """Elimina filas duplicadas del DataFrame."""
    return df.drop_duplicates()

def preprocesamiento_completo(ruta_archivo, 
                              columnas_numericas=None, 
                              columnas_categoricas=None,
                              estrategia_nulos='media'):
    """Ejecuta el pipeline completo de preprocesamiento."""
    df = cargar_datos(ruta_archivo)
    df = manejar_nulos(df, estrategia=estrategia_nulos)
    df = eliminar_duplicados(df)
    
    if columnas_numericas:
        df = normalizar_datos(df, columnas_numericas)
    
    if columnas_categoricas:
        df = codificar_categoricas(df, columnas_categoricas)
    
    return df

if __name__ == "__main__":
    # Ejemplo de uso
    datos_ejemplo = pd.DataFrame({
        'edad': [25, 30, None, 35, 30],
        'ciudad': ['Quito', 'Guayaquil', 'Cuenca', 'Quito', 'Guayaquil'],
        'salario': [45000, 52000, 48000, None, 52000]
    })
    
    datos_ejemplo.to_csv('datos_ejemplo.csv', index=False)
    
    df_procesado = preprocesamiento_completo(
        'datos_ejemplo.csv',
        columnas_numericas=['edad', 'salario'],
        columnas_categoricas=['ciudad'],
        estrategia_nulos='media'
    )
    
    print("Datos originales:")
    print(pd.read_csv('datos_ejemplo.csv'))
    print("\nDatos procesados:")
    print(df_procesado)
