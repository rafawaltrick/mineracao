from data_loading import load_data
from exploratory_analysis import descriptive_statistics, frequency_distribution, plot_gravity_distribution, correlation_heatmap
from preprocessing import handle_missing_values, remove_outliers, encode_categorical, normalize_numeric
from data_mining import split_data, train_model, evaluate_model
from visualization import plot_confusion_matrix

import pandas as pd

FILE_PATH = 'arquivo/si_env-2020.csv'

#Carregamento dos dados
df = load_data('arquivo/si_env-2020.csv')
df.columns = df.columns.str.strip()
df.columns = df.columns.str.lower()

print(df.columns.tolist())

# Columns
# ['num_boletim', 'data_hora_boletim', 'nº_envolvido', 'condutor', 'cod_severidade', 'desc_severidade', 'sexo', 'cinto_seguranca', 'embreagues', 'idade', 'nascimento', 'categoria_habilitacao', 'descricao_habilitacao', 'declaracao_obito', 'cod_severidade_antiga', 'especie_veiculo', 'pedestre', 'passageiro']

df.rename(columns={'nº_envolvido': 'numero_envolvido'}, inplace=True)


def convert_to_datetime(df):
    df['data_hora_boletim'] = pd.to_datetime(df['data_hora_boletim'], format='%d/%m/%Y %H:%M', errors='coerce')
    df['data_hora_boletim'] = df['data_hora_boletim'].dt.strftime('%Y-%m-%d %H:%M:%S')
    breakpoint()
    return df

convert_to_datetime(df)

# Análise Exploratória
descriptive_statistics(df)
frequency_distribution(df, ['condutor', 'cod_severidade', 'sexo', 'cinto_seguranca', 'embreagues', 'declaracao_obito'])
#plot_gravity_distribution(df)
correlation_heatmap(df)

# Pré-processamento
handle_missing_values(df)
df = remove_outliers(df)
df = encode_categorical(df, ['Condutor Responsável', 'Gravidade do Acidente', 'Sexo', 'Uso do Cinto de Segurança', 'Etilômetro', 'Óbito'])
df = normalize_numeric(df)

# Data Mining
X_train, X_test, y_train, y_test = split_data(df)
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)

# Visualização dos Resultados
y_pred = model.predict(X_test)
plot_confusion_matrix(y_test, y_pred)
