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
# (b) Las 10 películas con mayor ingreso (revenue)
# -----------------------------------------------------------
top_revenue_movies = df.nlargest(10, "revenue")[["title", "revenue"]].dropna()

top_revenue_movies["revenue"] = top_revenue_movies["revenue"].apply(lambda x: f"${x:,.0f}")

print("\n(b) 💰 Top 10 películas con mayor ingreso:")
print(top_revenue_movies.to_string(index=False))

# Gráfico de barras horizontales
plt.figure(figsize=(8, 6))
plt.barh(top_revenue_movies["title"], df.nlargest(10, "revenue")["revenue"], color="gold")
plt.xlabel("Ingresos (Billones de USD)")
plt.title("Top 10 películas con mayor ingreso")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# (d) Peor película de acuerdo a los votos de los usuarios
# -----------------------------------------------------------
worst_movies = df.nsmallest(10, "voteAvg")[["title", "voteAvg", "voteCount"]]

print("\n(d) ❌ Top 10 peores películas según los votos de los usuarios:")
print(worst_movies)

# Gráfico de barras horizontales
plt.figure(figsize=(8, 6))
plt.barh(worst_movies["title"], worst_movies["voteAvg"], color="red")
plt.xlabel("Promedio de Votos")
plt.title("Top 10 peores películas según los usuarios")
plt.xlim(0, 10)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# (f) Género principal de las 20 películas más recientes
# -----------------------------------------------------------
recent_movies = df.sort_values(by="releaseDate", ascending=False).head(20)
recent_movies["genre_main"] = recent_movies["genres"].str.split("|").str[0]

print("\n(f) 🎬 Género de las 20 películas más recientes:")
print(recent_movies[["title", "releaseDate", "genre_main"]])

# -----------------------------------------------------------
# (f) Género principal que predomina en el conjunto de datos
# -----------------------------------------------------------
df["genre_main"] = df["genres"].str.split("|").str[0]
genre_counts = df["genre_main"].value_counts()

print("\n📊 Género principal más frecuente en todo el dataset:")
print(genre_counts.head(10))

# Gráfico de barras de la distribución de géneros
plt.figure(figsize=(10, 5))
genre_counts.head(10).plot(kind="bar", color="skyblue")
plt.xlabel("Género")
plt.ylabel("Cantidad de Películas")
plt.title("Top 10 Géneros más frecuentes en el dataset")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# (f) Género de las películas más largas
# -----------------------------------------------------------
longest_movies = df.nlargest(10, "runtime")[["title", "runtime", "genre_main"]]
print("\n🎥 Género principal de las películas más largas:")
print(longest_movies)

# Gráfico de barras horizontales
plt.figure(figsize=(8, 5))
plt.barh(longest_movies["title"], longest_movies["runtime"], color="lightcoral")
plt.xlabel("Duración (minutos)")
plt.title("Top 10 Películas más largas")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()