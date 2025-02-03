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

# -----------------------------------------------------------
# (i) Influencia del reparto (hombres/mujeres) en popularidad e ingresos
# -----------------------------------------------------------
# Analizaremos la cantidad de mujeres en el elenco vs popularidad e ingresos
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1) DEFINIR RANGOS PARA AGRUPAR
women_bins = [0, 2, 5, 10, 20, 50, 200]  # Ajusta según tu dataset
women_labels = ["0-2", "3-5", "6-10", "11-20", "21-50", "50+"]

men_bins = [0, 2, 5, 10, 20, 50, 200]
men_labels = ["0-2", "3-5", "6-10", "11-20", "21-50", "50+"]

df["castWomenRange"] = pd.cut(df["castWomenAmount"], bins=women_bins, labels=women_labels)
df["castMenRange"] = pd.cut(df["castMenAmount"], bins=men_bins, labels=men_labels)

# ------------------------------------------------------------------------------
# 2) CALCULAR PROMEDIOS DE POPULARIDAD E INGRESOS POR CADA RANGO
# ------------------------------------------------------------------------------
women_popularity_mean = df.groupby("castWomenRange")["popularity"].mean()
women_revenue_mean = df.groupby("castWomenRange")["revenue"].mean()

men_popularity_mean = df.groupby("castMenRange")["popularity"].mean()
men_revenue_mean = df.groupby("castMenRange")["revenue"].mean()

print("Promedio de popularidad por rango de actrices:\n", women_popularity_mean, "\n")
print("Promedio de ingresos por rango de actrices:\n", women_revenue_mean, "\n")

print("Promedio de popularidad por rango de actores:\n", men_popularity_mean, "\n")
print("Promedio de ingresos por rango de actores:\n", men_revenue_mean, "\n")

# ------------------------------------------------------------------------------
# 3) GRÁFICOS DE BARRAS PARA VISUALIZAR ESOS PROMEDIOS
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Popularidad por rango de actrices
sns.barplot(
    ax=axes[0, 0],
    x=women_popularity_mean.index,
    y=women_popularity_mean.values,
    palette="Blues"
)
axes[0, 0].set_title("Popularidad promedio por rango de actrices")
axes[0, 0].set_xlabel("Rango de actrices")
axes[0, 0].set_ylabel("Popularidad promedio")

# (b) Ingresos por rango de actrices
sns.barplot(
    ax=axes[0, 1],
    x=women_revenue_mean.index,
    y=women_revenue_mean.values,
    palette="Greens"
)
axes[0, 1].set_title("Ingresos promedio por rango de actrices")
axes[0, 1].set_xlabel("Rango de actrices")
axes[0, 1].set_ylabel("Ingresos promedio (USD)")

# (c) Popularidad por rango de actores
sns.barplot(
    ax=axes[1, 0],
    x=men_popularity_mean.index,
    y=men_popularity_mean.values,
    palette="Blues"
)
axes[1, 0].set_title("Popularidad promedio por rango de actores")
axes[1, 0].set_xlabel("Rango de actores")
axes[1, 0].set_ylabel("Popularidad promedio")

# (d) Ingresos por rango de actores
sns.barplot(
    ax=axes[1, 1],
    x=men_revenue_mean.index,
    y=men_revenue_mean.values,
    palette="Greens"
)
axes[1, 1].set_title("Ingresos promedio por rango de actores")
axes[1, 1].set_xlabel("Rango de actores")
axes[1, 1].set_ylabel("Ingresos promedio (USD)")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 4) CALCULAR CORRELACIONES
#    Para cuantificar si existe relación lineal (positiva/negativa) y cuán fuerte.
# ------------------------------------------------------------------------------

corr_women_popularity = df["castWomenAmount"].corr(df["popularity"])
corr_women_revenue = df["castWomenAmount"].corr(df["revenue"])

corr_men_popularity = df["castMenAmount"].corr(df["popularity"])
corr_men_revenue = df["castMenAmount"].corr(df["revenue"])

print(f"Correlación (cantidad de actrices vs popularidad): {corr_women_popularity:.3f}")
print(f"Correlación (cantidad de actrices vs ingresos):   {corr_women_revenue:.3f}")
print(f"Correlación (cantidad de actores vs popularidad): {corr_men_popularity:.3f}")
print(f"Correlación (cantidad de actores vs ingresos):   {corr_men_revenue:.3f}")



# -----------------------------------------------------------
# (k) Relación entre presupuesto e ingresos (histograma y diagrama de dispersión)
# -----------------------------------------------------------

df["budget_millions"] = df["budget"] / 1_000_000
df["revenue_millions"] = df["revenue"] / 1_000_000

# 1) Diagrama de dispersión
plt.figure(figsize=(8, 6))
plt.scatter(df["budget_millions"], df["revenue_millions"], alpha=0.5, color="purple")
plt.xlabel("Presupuesto (Millones USD)")
plt.ylabel("Ingresos (Millones USD)")
plt.title("Relación entre Presupuesto e Ingresos (en millones)")
plt.tight_layout()
plt.show()

# 2) Histograma de la diferencia (o de la propia variable)
#    Por ejemplo, histograma del presupuesto
plt.figure(figsize=(8, 5))
plt.hist(df["budget_millions"].dropna(), bins=50, color="teal", edgecolor="black")
plt.xlim(0, 200)  
plt.xlabel("Presupuesto (Millones USD)")
plt.ylabel("Frecuencia")
plt.title("Distribución del Presupuesto (en millones)")
plt.tight_layout()
plt.show()
