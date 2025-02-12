import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def analyze_and_predict(data_path, year_filter=None):
    # Cargar datos
    data = pd.read_csv(data_path)

    # Limpieza de datos
    data[' gdp_for_year ($) '] = data[' gdp_for_year ($) '].str.replace(',', '').astype(float)
    data['suicides_no'] = data['suicides_no'].fillna(0)
    data['HDI for year'] = data['HDI for year'].fillna(data['HDI for year'].mean())

    # Filtrar datos por año (si se proporciona)
    if year_filter:
        data = data[data['year'] == year_filter]
        if data.empty:
            print(f"No se encontraron datos para el año {year_filter}.")
            return

    # Análisis por género
    gender_counts = data.groupby('sex')['suicides_no'].sum()
    print(f"\nTotal de suicidios por género en el año {year_filter if year_filter else 'todos los años'}:")
    print(gender_counts)

    # Análisis por generación
    generation_counts = data.groupby('generation')['suicides_no'].sum()
    print(f"\nTotal de suicidios por generación en el año {year_filter if year_filter else 'todos los años'}:")
    print(generation_counts)

    # Relación PIB per cápita y tasa de suicidios
    print("\nRelación PIB per cápita y tasa de suicidios (descripción estadística):")
    print(data[['gdp_per_capita ($)', 'suicides/100k pop']].describe())

    # Relación entre género y rango de edad
    age_gender = data.groupby(['sex', 'age'])['suicides_no'].sum().unstack()

    # Imprimir los datos en consola
    print("\nDistribución de suicidios por género y rango de edad:")
    print(age_gender)

    # Generar gráfica de relación entre género y rango de edad
    age_gender.plot(kind='bar', stacked=True, figsize=(10, 6), 
                    title='Relación entre género y rango de edad en suicidios')
    plt.ylabel('Número de suicidios', fontsize=12)
    plt.xlabel('Género', fontsize=12)
    plt.legend(title='Rango de edad', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Gráfico: Relación PIB per cápita y tasa de suicidios
    plt.figure(figsize=(10, 6))
    plt.scatter(data['gdp_per_capita ($)'], data['suicides/100k pop'], alpha=0.5, color='purple', edgecolors='black')
    plt.title(f'Relación entre PIB per cápita y tasa de suicidios ({year_filter if year_filter else "todos los años"})', fontsize=14)
    plt.xlabel('PIB per cápita ($)', fontsize=12)
    plt.ylabel('Tasa de suicidios (por 100k habitantes)', fontsize=12)
    plt.grid(True)
    plt.show()

    # Gráfico: Distribución por género
    gender_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ylabel='', 
                        title=f'Distribución de suicidios por género ({year_filter})' if year_filter else 'Distribución por género (todos los años)',
                        figsize=(6, 6), colors=['#1f77b4', '#ff7f0e'])
    plt.show()

    # Gráfico: Distribución por generación
    generation_counts.plot(kind='bar', color='green', edgecolor='black')
    plt.title(f'Distribución de suicidios por generación ({year_filter})' if year_filter else 'Distribución por generación (todos los años)', fontsize=14)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Número de suicidios', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Cálculo de correlaciones
    numeric_columns = ['year', 'suicides_no', 'population', 'suicides/100k pop', 
                        ' gdp_for_year ($) ', 'gdp_per_capita ($)', 'HDI for year'] + \
                        [col for col in data.columns if col.startswith(('country_', 'sex_', 'age_', 'generation_'))]

    correlations = data[numeric_columns].corr()

    # Correlación con suicidios
    pearson_corr_suicides = correlations['suicides_no'].drop('suicides_no')
    print("\nCorrelación de Pearson con el número de suicidios:")
    print(pearson_corr_suicides)

    # Evaluar influencia con base en correlación
    for var, corr in pearson_corr_suicides.items():
        if abs(corr) > 0.5:
            print(f"La variable '{var}' tiene una influencia fuerte (correlación de {corr:.2f}) sobre los suicidios.")
        elif abs(corr) > 0.3:
            print(f"La variable '{var}' tiene una influencia moderada (correlación de {corr:.2f}) sobre los suicidios.")
        else:
            print(f"La variable '{var}' no tiene una influencia significativa (correlación de {corr:.2f}) sobre los suicidios.")

    # Preparación de datos para el modelo
    data_encoded = pd.get_dummies(data, columns=['sex', 'age', 'generation'], drop_first=True)
    X = data_encoded[['gdp_per_capita ($)', 'HDI for year', 'year'] + 
                    [col for col in data_encoded.columns if col.startswith(('sex_', 'age_', 'generation_'))]]
    y = data_encoded['suicides_no']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Métricas de evaluación
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nError Cuadrático Medio (MSE): {mse:.2f}")
    print(f"Coeficiente de Determinación (R²): {r2:.2f}")

    # Gráfico: Valores reales vs Predicciones
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Valores Reales', color='blue', linestyle='-', marker='o', alpha=0.7)
    plt.plot(y_pred, label='Predicciones', color='red', linestyle='--', marker='x', alpha=0.7)
    plt.title('Predicciones vs Valores Reales', fontsize=14)
    plt.xlabel('Índice de Muestra', fontsize=12)
    plt.ylabel('Número de Suicidios', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, mse, r2

# Ejecutar el análisis y predicción
if __name__ == "__main__":
    data_path = "dataset.csv"  
    year_to_filter = input("Indica el año que deseas analizar (o presiona Enter para analizar todos los años): ")
    year_to_filter = int(year_to_filter) if year_to_filter.isdigit() else None
    model, mse, r2 = analyze_and_predict(data_path, year_filter=year_to_filter)
