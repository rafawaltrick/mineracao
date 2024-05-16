import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def descriptive_statistics(df):
    print(df.describe(include='all'))

def frequency_distribution(df, categorical_columns):
    for col in categorical_columns:
        print(f'Distribuição de {col}:')
        print(df[col].value_counts())
        print()

def plot_gravity_distribution(df):
    sns.countplot(x='Gravidade do Acidente', data=df)
    plt.title('Distribuição de Gravidade dos Acidentes')
    plt.show()

def correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.title('Mapa de Calor de Correlação')
    plt.show()
