import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------
# Cargar el dataset limpio
# -----------------------------------------------------------
data_path = "data/movies_clean.csv"
df = pd.read_csv(data_path)

# Convertir 'releaseDate' a tipo fecha
df["releaseDate"] = pd.to_datetime(df["releaseDate"], errors="coerce")

# -----------------------------------------------------------
# Conversión de tipos para evitar errores en cálculos
# -----------------------------------------------------------
numeric_cols = ["budget", "revenue", "voteCount", "popularity",
                "castWomenAmount", "castMenAmount", "actorsAmount"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Crear columna adicional para el año y el mes de lanzamiento
df["year"] = df["releaseDate"].dt.year
df["month"] = df["releaseDate"].dt.month  # 1 = Enero, 12 = Diciembre

# Crear la columna 'profit' (ganancia = ingresos - presupuesto)
df["profit"] = df["revenue"] - df["budget"]

# -----------------------------------------------------------
# (a) Las 10 películas con mayor presupuesto
# -----------------------------------------------------------
top_budget_movies = df.nlargest(10, "budget")[["title", "budget"]].dropna()

print("\n(a) 🎬 Top 10 películas con mayor presupuesto:")
print(top_budget_movies)

# Gráfico de barras horizontales
plt.figure(figsize=(8, 6))
plt.barh(top_budget_movies["title"], top_budget_movies["budget"], color="skyblue")
plt.xlabel("Presupuesto (USD)")
plt.title("Top 10 películas con mayor presupuesto")
plt.gca().invert_yaxis()  # Invertir para que la más costosa aparezca arriba
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# (c) La película con más votos
# -----------------------------------------------------------
# Obtenemos el índice de la fila con la máxima 'voteCount'
most_voted_movie = df.loc[df["voteCount"].idxmax(), ["title", "voteCount"]]

print("\n(c) 🏆 Película con más votos:")
print(most_voted_movie)

# (Opcional) Podrías mostrar también el top 5
top_5_voted = df.nlargest(5, "voteCount")[["title", "voteCount"]]
print("\nTop 5 películas con más votos:")
print(top_5_voted)

# Gráfico (opcional)
plt.figure(figsize=(8, 5))
plt.bar(top_5_voted["title"], top_5_voted["voteCount"], color="orange")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Título")
plt.ylabel("Cantidad de votos")
plt.title("Top 5 películas con más votos")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# (e) Cuántas películas se hicieron por año (gráfico de barras)
# -----------------------------------------------------------
movies_per_year = df["year"].value_counts().sort_index()
print("\n(e) 📅 Número de películas por año:")
print(movies_per_year)

plt.figure(figsize=(12, 6))
plt.bar(movies_per_year.index, movies_per_year.values, color="lightgreen")
plt.xlabel("Año")
plt.ylabel("Número de películas")
plt.title("Número de películas producidas por año")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# (g) Los géneros que generaron más ganancias
# -----------------------------------------------------------
# 'genres' está separada por '|', la convertimos a lista y hacemos explode
df["genres"] = df["genres"].fillna("")
df["genres"] = df["genres"].apply(lambda x: x.split("|") if isinstance(x, str) else [])

genres_profit = (
    df.explode("genres")
      .groupby("genres")["profit"]
      .sum()
      .sort_values(ascending=False)
)

print("\n(g) 💰 Géneros con más ganancias (totales):")
print(genres_profit.head(10))

# Gráfico de barras para los 10 géneros con mayores ganancias
plt.figure(figsize=(10, 6))
genres_profit.head(10).plot(kind='bar', color="green")
plt.xlabel("Género")
plt.ylabel("Ganancia total (USD)")
plt.title("Top 10 géneros con más ganancias")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()