from sklearn.calibration import LabelEncoder
from sklearn.cluster import KMeans 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import StandardScaler
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


# COLUNAS PRINCIPAIS
# ['num_boletim', 'data_hora_boletim', 'numero_envolvido', 'condutor', 'cod_severidade', 'sexo', 'cinto_seguranca', 'embreagues', 'idade', 'declaracao_obito', 'especie_veiculo']


## Associar condutor com idade

df.rename(columns={'nº_envolvido': 'numero_envolvido'}, inplace=True)
df = df.drop(columns=['desc_severidade', 'nascimento', 'categoria_habilitacao', 'descricao_habilitacao', 'cod_severidade_antiga', 'passageiro', 'pedestre'])

def convert_to_datetime(df):
    df['data_hora_boletim'] = pd.to_datetime(df['data_hora_boletim'], format='%d/%m/%Y %H:%M', errors='coerce')
    df['data_hora_boletim'] = df['data_hora_boletim'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

def parse_sex_column_to_bool(df):
    df['sexo'] = df['sexo'].map({'M': 1, 'F': 0})
    most_frequent = df['sexo'].mode()[0]
    df['sexo'] = df['sexo'].fillna(most_frequent).astype(int)
    return df

def convert_driver_to_bool(df):
    df['condutor'] = df['condutor'].map({'S': 1, 'N': 0})
    most_frequent = df['condutor'].mode()[0]
    df['condutor'] = df['condutor'].fillna(most_frequent).astype(int)
    return df

def convert_seat_belt_to_bool(df):
    df['cinto_seguranca'] = df['cinto_seguranca'].map({'SIM': 1, 'NÃO': 0})
    most_frequent = df['cinto_seguranca'].mode()[0]
    df['cinto_seguranca'] = df['cinto_seguranca'].fillna(most_frequent).astype(int)
    return df

def convert_drunkenness_to_bool(df):
    df['embreagues'] = df['embreagues'].map({'SIM': 1, 'NÃO': 0})
    most_frequent = df['embreagues'].mode()[0]
    df['embreagues'] = df['embreagues'].fillna(most_frequent).astype(int)
    return df

def evaluate_age_range(df):
    bins = [18, 24, 38, 48, 55, 65, 75, float('inf')]
    labels = ["18-24", "25-38", "39-48", "49-55", "56-65", "66-75", "76+"]
    df['faixa_etaria'] = pd.cut(df['idade'], bins=bins, labels=labels, right=False)
    df = df.drop(columns=['idade'])
    return df


def evaluate_type_veicle_by_drive_license(df):
    df['especie_veiculo'] = df['especie_veiculo'].str.strip()
    df['especie_veiculo'] = df['especie_veiculo'].str.replace(" ", "-")
    cnh_categories = {
        'BICICLETA': 'CAT-A',
        'MOTOCICLETA': 'CAT-A',
        'MOTONETA': 'CAT-A',
        'CICLOMOTOR': 'CAT-A',
        'TRICICLO': 'CAT-A',
        'AUTOMOVEL': 'CAT-B',
        'CAMIONETA': 'CAT-B',
        'KOMBI': 'CAT-B',
        'CARRO-DE-MAO': 'CAT-OUTROS',
        'TRATOR-DE-RODAS': 'CAT-OUTROS',
        'CARROCA': 'CAT-OUTROS',
        'PATINETE': 'CAT-OUTROS',
        'CAMINHAO': 'CAT-C',
        'CAMINHONETE': 'CAT-C',
        'CAMINHAO-TRATOR': 'CAT-C',
        'REBOQUE-E-SEMI-REBOQUE': 'CAT-C',
        'TRATOR-MISTO': 'CAT-C',
        'ONIBUS': 'CAT-D',
        'MICROONIBUS': 'CAT-D',
        'BONDE': 'CAT-D',
        '':'CAT-NI', #NÃO INFORMADO
        'NAO-INFORMADO': 'CAT-NI' #NÃO INFORMADO
    }
    df['especie_veiculo'] = df['especie_veiculo'].fillna("")
    df['categoria_cnh'] = df['especie_veiculo'].map(cnh_categories)
    df = df.drop(columns=['especie_veiculo'])
    return df

def convert_severity(df):
    # df['cod_severidade'] = df['cod_severidade'].to_string(index=False).strip().replace('\n','')
    severity_categories = {
        1: 'NAO FATAL',
        2: 'FATAL',
        3: 'SEM FERIMENTOS',
        0: 'NAO INFORMADO'
    }
    df['cod_severidade'] = df['cod_severidade'].fillna(0)
    df['cod_severidade'] = df['cod_severidade'].map(severity_categories)
    return df

def training_classification_model(df):
    label_encoders = {}
    for column in ['cod_severidade', 'faixa_etaria', 'categoria_cnh']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le


    # Seleção das colunas relevantes
    X = df[[ 'condutor', 'cinto_seguranca', 'embreagues', 'declaracao_obito']]
    y = df['cod_severidade']

    # Normalização dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.6, random_state=42)
    print(len(X_scaled))

    # Aplicação do K-means
    knn = KNeighborsClassifier(n_neighbors=3,  weights='distance', algorithm='auto')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Avaliação do modelo
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Adicionando a previsão ao DataFrame original para análise
    df['pred_severidade'] = knn.predict(scaler.transform(X))
    df['pred_severidade'] = label_encoders['cod_severidade'].inverse_transform(df['pred_severidade'])






df = convert_to_datetime(df)
df = parse_sex_column_to_bool(df)
df = convert_driver_to_bool(df)
df = convert_seat_belt_to_bool(df)
df = convert_drunkenness_to_bool(df)
df = evaluate_age_range(df)
df = evaluate_type_veicle_by_drive_license(df)
df = convert_severity(df)


print(df)
print(df.columns.tolist())
print(df.dtypes)
print(df.head(20))

training_classification_model(df)
print(df.head(10))

# Análise Exploratória
# descriptive_statistics(df)
# frequency_distribution(df, ['condutor', 'cod_severidade', 'sexo', 'cinto_seguranca', 'embreagues', 'declaracao_obito'])
# #plot_gravity_distribution(df)
# correlation_heatmap(df)

# Pré-processamento

#handle_missing_values(df)
# df = remove_outliers(df)
# df = encode_categorical(df, ['condutor', 'cod_severidade', 'sexo', 'cinto_seguranca', 'embreagues', 'declaracao_obito'])
# df = normalize_numeric(df)


# # Data Mining
# X_train, X_test, y_train, y_test = split_data(df)
# model = train_model(X_train, y_train)
# evaluate_model(model, X_test, y_test)

# # Visualização dos Resultados
# y_pred = model.predict(X_test)
# plot_confusion_matrix(y_test, y_pred)
