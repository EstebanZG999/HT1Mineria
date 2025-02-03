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
plt.figure(figsize=(7, 4)) 
plt.barh(worst_movies["title"], worst_movies["voteAvg"], color="red")
plt.xlabel("Promedio de Votos")
plt.title("Top 10 peores películas según los usuarios")
plt.xlim(0, 5)
plt.gca().invert_yaxis()
plt.xticks(fontsize=9)
plt.yticks(fontsize=7)
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

# -----------------------------------------------------------
# (h) ¿La cantidad de actores influye en los ingresos?
# -----------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="actorsAmount", y="revenue", alpha=0.5)
plt.xlabel("Cantidad de Actores")
plt.ylabel("Ingresos (USD)")
plt.title("Relación entre la cantidad de actores y los ingresos")
plt.show()

# Calcular correlación entre actores y ingresos
correlation = df[["actorsAmount", "revenue"]].corr().iloc[0, 1]
print(f"\n📊 Correlación entre cantidad de actores e ingresos: {correlation:.2f}")

# -----------------------------------------------------------
# (h) ¿Se han hecho películas con más actores en los últimos años?
# -----------------------------------------------------------
avg_actors_per_year = df.groupby("year")["actorsAmount"].mean()

plt.figure(figsize=(10, 5))
plt.plot(avg_actors_per_year.index, avg_actors_per_year.values, marker="o", linestyle="-", color="purple")
plt.xlabel("Año")
plt.ylabel("Promedio de Actores por Película")
plt.title("Evolución del número de actores en las películas")
plt.grid()
plt.show()

# -----------------------------------------------------------
# (j) Obtener las 20 películas mejor calificadas
# -----------------------------------------------------------
top_rated_movies = df.nlargest(20, "voteAvg")[["title", "voteAvg", "director"]].dropna()
top_rated_movies["director"] = top_rated_movies["director"].apply(lambda x: x if len(x) <= 30 else x[:27] + "...")

print("\n(g) 🎬 Directores de las 20 películas mejor calificadas:")
print(top_rated_movies.to_string(index=False))

director_counts = top_rated_movies["director"].value_counts()

# Gráfico de los directores con más películas en el Top 20
plt.figure(figsize=(8, 4))
director_counts.plot(kind="bar", color="royalblue")
plt.xlabel("Director")
plt.ylabel("Cantidad de películas en el Top 20")
plt.title("Directores con más películas mejor calificadas")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# (l) ¿Se asocian ciertos meses de lanzamiento con mejores ingresos?
# -----------------------------------------------------------

monthly_revenue = df.groupby("month")["revenue"].mean().sort_index()
formatted_revenue = monthly_revenue.apply(lambda x: f"${x:,.0f}")

print("\n📅 Promedio de ingresos por mes:")
print(formatted_revenue)

# Gráfico de barras
plt.figure(figsize=(10, 5))
plt.bar(monthly_revenue.index, monthly_revenue.values, color="royalblue")
plt.xlabel("Mes de lanzamiento")
plt.ylabel("Ingreso promedio (Millones de USD)")
plt.xticks(range(1, 13), ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"], rotation=45)
plt.title("Promedio de ingresos por mes de lanzamiento")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# (n) Correlación entre calificaciones y éxito comercial
# -----------------------------------------------------------
correlation = df[["voteAvg", "revenue"]].corr().iloc[0, 1]

print(f"\n⭐ Correlación entre calificaciones y éxito comercial: {correlation:.2f}")

# Gráfico de dispersión
plt.figure(figsize=(8, 6))
plt.scatter(df["voteAvg"], df["revenue"], alpha=0.5)
plt.xlabel("Calificación Promedio (voteAvg)")
plt.ylabel("Ingresos (USD)")
plt.title("Relación entre Calificaciones y Éxito Comercial")
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------
# (p) ¿Popularidad del elenco directamente correlacionada con el éxito?
# ----------------------------------------------------------------------
# Convertir 'actorsPopularity' en valores numéricos (promedio de la lista)
def parse_and_average(popularity_str):
    try:
        values = list(map(float, popularity_str.split("|")))  # Convertir cada número a float
        return np.mean(values) if values else np.nan  # Calcular el promedio
    except:
        return np.nan  # Si hay un error, devolver NaN

# Aplicar la conversión a la columna 'actorsPopularity'
df["actorsPopularity"] = df["actorsPopularity"].astype(str).apply(parse_and_average)

# Calcular la correlación
correlation_cast_popularity = df["actorsPopularity"].corr(df["revenue"])

print(f"\n🎭 Correlación entre popularidad del elenco y éxito de taquilla: {correlation_cast_popularity:.2f}")

# Gráfico de dispersión
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["actorsPopularity"], y=df["revenue"], alpha=0.5)
plt.xlabel("Popularidad del Elenco (Promedio de actorsPopularity)")
plt.ylabel("Ingresos (USD)")
plt.title("Relación entre Popularidad del Elenco y Éxito de Taquilla")
plt.show()
