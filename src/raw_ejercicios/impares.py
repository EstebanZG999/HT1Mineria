import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

df["profit"] = df["revenue"] - df["budget"]


# Para mostrar todos los valores monetarios en millones
df["budget_millions"] = df["budget"] / 1_000_000
df["revenue_millions"] = df["revenue"] / 1_000_000
df["profit_millions"] = df["profit"] / 1_000_000

# -----------------------------------------------------------
# (a) Las 10 películas con mayor presupuesto
# -----------------------------------------------------------
top_budget_movies = df.nlargest(10, "budget_millions")[["title", "budget_millions"]].dropna()

print("\n(a) 🎬 Top 10 películas con mayor presupuesto (en millones):")
print(top_budget_movies)

# Gráfico de barras horizontales
plt.figure(figsize=(8, 6))
plt.barh(top_budget_movies["title"], top_budget_movies["budget_millions"], color="skyblue")
plt.xlabel("Presupuesto (Millones USD)")
plt.title("Top 10 películas con mayor presupuesto (millones)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# (c) La película con más votos
# -----------------------------------------------------------
# Obtenemos el índice de la fila con la máxima 'voteCount'
most_voted_movie = df.loc[df["voteCount"].idxmax(), ["title", "voteCount"]]
print("\n(c) 🏆 Película con más votos:")
print(most_voted_movie)

# mostrar también el top 5
top_5_voted = df.nlargest(5, "voteCount")[["title", "voteCount"]]
print("\nTop 5 películas con más votos:")
print(top_5_voted)

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
df_1960 = df[df["year"] >= 1960].copy()

movies_per_year_1960 = df_1960["year"].value_counts().sort_index()
print("\n(e) 📅 Número de películas por año (desde 1960):")
print(movies_per_year_1960)

plt.figure(figsize=(12, 6))
plt.bar(movies_per_year_1960.index, movies_per_year_1960.values, color="lightgreen")
plt.xlabel("Año (>= 1960)")
plt.ylabel("Número de películas")
plt.title("Número de películas producidas por año (desde 1960)")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# (g) Los géneros que generaron más ganancias
# -----------------------------------------------------------
df["genres"] = df["genres"].fillna("")
df["genres"] = df["genres"].apply(lambda x: x.split("|") if isinstance(x, str) else [])

genres_profit = (
    df.explode("genres")
      .groupby("genres")["profit_millions"]
      .sum()
      .sort_values(ascending=False)
)

print("\n(g) 💰 Géneros con más ganancias (totales, en millones):")
print(genres_profit.head(10))

# Gráfico de barras para los 10 géneros con mayores ganancias
plt.figure(figsize=(10, 6))
genres_profit.head(10).plot(kind='bar', color="green")
plt.xlabel("Género")
plt.ylabel("Ganancia total (Millones USD)")
plt.title("Top 10 géneros con más ganancias (en millones)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# (i) Influencia del reparto (hombres/mujeres) en popularidad e ingresos
# -----------------------------------------------------------

# 1) DEFINIR RANGOS PARA AGRUPAR
women_bins = [0, 2, 5, 10, 20, 50, 200]  
women_labels = ["0-2", "3-5", "6-10", "11-20", "21-50", "50+"]

men_bins = [0, 2, 5, 10, 20, 50, 200]
men_labels = ["0-2", "3-5", "6-10", "11-20", "21-50", "50+"]

df["castWomenRange"] = pd.cut(df["castWomenAmount"], bins=women_bins, labels=women_labels)
df["castMenRange"] = pd.cut(df["castMenAmount"], bins=men_bins, labels=men_labels)

# ------------------------------------------------------------------------------
# 2) CALCULAR PROMEDIOS DE POPULARIDAD E INGRESOS POR CADA RANGO
# ------------------------------------------------------------------------------
women_popularity_mean = df.groupby("castWomenRange")["popularity"].mean()
women_revenue_mean = df.groupby("castWomenRange")["revenue_millions"].mean()

men_popularity_mean = df.groupby("castMenRange")["popularity"].mean()
men_revenue_mean = df.groupby("castMenRange")["revenue_millions"].mean()

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
axes[0, 1].set_title("Ingresos promedio (millones) por rango de actrices")
axes[0, 1].set_xlabel("Rango de actrices")
axes[0, 1].set_ylabel("Ingresos promedio (M USD)")

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
axes[1, 1].set_title("Ingresos promedio (millones) por rango de actores")
axes[1, 1].set_xlabel("Rango de actores")
axes[1, 1].set_ylabel("Ingresos promedio (M USD)")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 4) CALCULAR CORRELACIONES
# ------------------------------------------------------------------------------

corr_women_popularity = df["castWomenAmount"].corr(df["popularity"])
corr_women_revenue = df["castWomenAmount"].corr(df["revenue_millions"])

corr_men_popularity = df["castMenAmount"].corr(df["popularity"])
corr_men_revenue = df["castMenAmount"].corr(df["revenue_millions"])

print(f"Correlación (cantidad de actrices vs. popularidad): {corr_women_popularity:.3f}")
print(f"Correlación (cantidad de actrices vs. ingresos MUSD): {corr_women_revenue:.3f}")
print(f"Correlación (cantidad de actores vs. popularidad): {corr_men_popularity:.3f}")
print(f"Correlación (cantidad de actores vs. ingresos MUSD): {corr_men_revenue:.3f}")


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
plt.figure(figsize=(8, 5))
plt.hist(df["budget_millions"].dropna(), bins=50, color="teal", edgecolor="black")
plt.xlim(0, 200)  
plt.xlabel("Presupuesto (Millones USD)")
plt.ylabel("Frecuencia")
plt.title("Distribución del Presupuesto (en millones)")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# (m) En qué meses se lanzaron las películas con mayores ingresos
#     y el promedio de ingresos por mes
# -----------------------------------------------------------
# 1) Calculamos el total o el promedio de ingresos por mes
revenue_by_month = df.groupby("month")["revenue_millions"].mean().sort_values(ascending=False)
print("\n(m) Meses con mayores ingresos (PROMEDIO, en millones):")
print(revenue_by_month)

# 2) Gráfico de barras para ver el promedio de ingresos por mes
plt.figure(figsize=(8, 5))
plt.bar(revenue_by_month.index, revenue_by_month.values, color="gold")
plt.xlabel("Mes de lanzamiento")
plt.ylabel("Ingresos promedio (Millones USD)")
plt.title("Promedio de ingresos (millones) por mes de lanzamiento")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()

top_income_movies = df.nlargest(50, "revenue_millions").dropna(subset=["month"])
count_month_top = top_income_movies["month"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(count_month_top.index, count_month_top.values, color="tomato")
plt.xlabel("Mes de lanzamiento")
plt.ylabel("Nº de películas (top 50 en ingresos)")
plt.title("Distribución de meses de lanzamiento en el top 50 de ingresos")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# (o) Estrategias de marketing (videos promocionales o páginas oficiales)
#     que generaron mejores resultados.
# -----------------------------------------------------------
# 1) Si no existe la columna booleana, la creamos:
df["has_homepage"] = ~df["homePage"].isna()

# 2) Agrupar y calcular promedios
marketing_video_revenue = df.groupby("video")["revenue_millions"].mean()
marketing_video_popularity = df.groupby("video")["popularity"].mean()

marketing_homepage_revenue = df.groupby("has_homepage")["revenue_millions"].mean()
marketing_homepage_popularity = df.groupby("has_homepage")["popularity"].mean()

print("\nPromedio de ingresos (millones) según 'video':\n", marketing_video_revenue)
print("\nPromedio de popularidad según 'video':\n", marketing_video_popularity)

print("\nPromedio de ingresos (millones) según 'has_homepage':\n", marketing_homepage_revenue)
print("\nPromedio de popularidad según 'has_homepage':\n", marketing_homepage_popularity)

# 3) Gráficos de barras para ver de forma clara
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Ingresos vs. video
sns.barplot(
    ax=axes[0, 0],
    x=marketing_video_revenue.index.astype(str),  
    y=marketing_video_revenue.values,
    palette="Set2"
)
axes[0, 0].set_title("Ingresos promedio (millones) según 'video'")
axes[0, 0].set_xlabel("¿Tiene video promocional?")
axes[0, 0].set_ylabel("Ingresos promedio (M USD)")

# (b) Popularidad vs. video
sns.barplot(
    ax=axes[0, 1],
    x=marketing_video_popularity.index.astype(str),
    y=marketing_video_popularity.values,
    palette="Set2"
)
axes[0, 1].set_title("Popularidad promedio según 'video'")
axes[0, 1].set_xlabel("¿Tiene video promocional?")
axes[0, 1].set_ylabel("Popularidad promedio")

# (c) Ingresos vs. has_homepage
sns.barplot(
    ax=axes[1, 0],
    x=marketing_homepage_revenue.index.astype(str),
    y=marketing_homepage_revenue.values,
    palette="Set2"
)
axes[1, 0].set_title("Ingresos promedio (millones) según 'has_homepage'")
axes[1, 0].set_xlabel("¿Tiene página oficial?")
axes[1, 0].set_ylabel("Ingresos promedio (M USD)")

# (d) Popularidad vs. has_homepage
sns.barplot(
    ax=axes[1, 1],
    x=marketing_homepage_popularity.index.astype(str),
    y=marketing_homepage_popularity.values,
    palette="Set2"
)
axes[1, 1].set_title("Popularidad promedio según 'has_homepage'")
axes[1, 1].set_xlabel("¿Tiene página oficial?")
axes[1, 1].set_ylabel("Popularidad promedio")

plt.tight_layout()
plt.show()

# Combinación (video + has_homepage)
combo_revenue = df.groupby(["video", "has_homepage"])["revenue_millions"].mean()
combo_popularity = df.groupby(["video", "has_homepage"])["popularity"].mean()

print("\nIngresos promedio (millones) por (video, has_homepage):\n", combo_revenue)
print("\nPopularidad promedio por (video, has_homepage):\n", combo_popularity)

print("\n✅ Análisis finalizado para preguntas (a, c, e, g, i, k, m, o).")