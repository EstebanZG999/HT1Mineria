import pandas as pd
import os

# Definir la ruta al archivo
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")  # Carpeta donde se guarda el archivo
data_path = os.path.join(data_dir, "movies.csv")  # Archivo de entrada
clean_data_path = os.path.join(data_dir, "movies_clean.csv")  # Archivo de salida

data_dir = os.path.normpath(data_dir)
data_path = os.path.normpath(data_path)
clean_data_path = os.path.normpath(clean_data_path)

print(f"📂 Ruta al archivo de entrada: {data_path}")
print(f"📂 Ruta al archivo de salida: {clean_data_path}")

# Verificar si el archivo existe
if not os.path.isfile(data_path):
    print("❌ El archivo movies.csv no se encuentra en la ruta especificada.")
else:
    print("✅ El archivo movies.csv ha sido encontrado correctamente.")

    # Cargar el dataset
    df = pd.read_csv(data_path, encoding="ISO-8859-1")

    # Convertir 'releaseDate' a tipo fecha
    df["releaseDate"] = pd.to_datetime(df["releaseDate"], errors="coerce")

    # Mostrar información general
    print("\n🔍 Información general del dataset:")
    print(df.info())

    # Revisar datos faltantes
    print("\n⚠️ Datos faltantes en el dataset:")
    print(df.isnull().sum())

    # Descripción estadística de las variables numéricas
    print("\n📊 Estadísticas de las variables numéricas:")
    print(df.describe())

    # 📌 **Crear la carpeta 'data/' si no existe**
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📂 Carpeta creada: {data_dir}")

    # Guardar el dataset limpio
    df.to_csv(clean_data_path, index=False)
    print(f"\n✅ Datos guardados en: {clean_data_path}")
