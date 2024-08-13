# Importação de bibliotecas.
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from itertools import product
from io import StringIO

# Pegando caminho da pasta automaticamente pela execução do arquivo.
parent_directory = os.path.dirname(os.path.abspath(__file__))
from utils import (bucket_name,
                   distances_key,
                   freight_costs_key,
                   destinations_key,
                   s3)

# Buscando os arquivos .csv na aws.
def get_aws_csv(key: str) -> pd.DataFrame:
    """Busca um arquivo csv na AWS.

    Parameters
    ----------
    key : str
        Chave para buscar na s3;

    Returns
    -------
    pd.DataFrame
        Dataframe especificado.
    """

    # Pegando o objeto.
    response = s3.get_object(Bucket=bucket_name,
                             Key=key)

    # Lendo o conteudo do objeto.
    csv = response['Body'].read().decode('utf-8')

    # Convertendo o csv em dataframe.
    dataframe = pd.read_csv(StringIO(csv), delimiter=';', decimal=',')

    return dataframe

def get_freight_costs_distances_by_destination(df_freight_costs: pd.DataFrame,
                                              df_distances: pd.DataFrame,
                                              df_destinations: pd.DataFrame) -> pd.DataFrame:
    """Custos dos fretes pelos destinos.

    Parameters
    ----------
    df_freight_costs : pd.Dataframe
        Custos dos fretes.
    df_distances : pd.Dataframe
        Distâncias.
    df_destination : pd.Dataframe
        Destinos.

    Returns
    -------
    pd.DataFrame
        Custo dos fretes, distância de acordo com os destinos especificados.
    """

    # Unindo as bases de cotações e distâncias.
    df_distances = df_distances.astype(float)
    df_freight_costs_distances = pd.merge(df_freight_costs,
                                        df_distances,
                                        on=['id_city_origin', 'id_city_destination'])
    
    # Lista de destinos especificados
    destinations = df_destinations['Destination'].values.astype(float).tolist()

    # Filtrando as cotações para os destinos especificados.
    df_freight_costs_distances = df_freight_costs_distances[df_freight_costs_distances['id_city_destination'].isin(destinations)]

    # Salvando o arquivo csv e gerando o primeiro Output.
    df_freight_costs_distances.to_csv(f'{parent_directory}/outputs/Output_1.csv')

    # Convertendo as datas de string para TimeStamp da biblioteca Pandas.
    df_freight_costs_distances['dt_reference'] = pd.to_datetime(df_freight_costs_distances['dt_reference'], format='%d/%m/%Y')
    df_freight_costs_distances.set_index('dt_reference', inplace=True)
    return df_freight_costs_distances

def generate_forecast(df_freight_costs_distances: pd.DataFrame,
                      origin:float,
                      destination:float,
                      n: int) -> pd.DataFrame:
    """Gera previsões de custo de frete.

    Parameters
    ----------
    df_freight_costs_distances : pd.DataFrame
        Custo e distância de fretes.
    origin : int
        Número de origem.
    n : int
        Número de destino.
    destination : int
        Número de semanas para previsão.

    Returns
    -------
    pd.DataFrame
        Previsão.
    """    

    # Rota para origem e destino,
    df_route = df_freight_costs_distances[
        (df_freight_costs_distances['id_city_origin'] == origin) & 
        (df_freight_costs_distances['id_city_destination'] == destination)
    ].sort_values(by='dt_reference')

    # Pula, caso a origem e destino tenham menos que 2 dados.
    if len(df_route) < 2:
        return

    # Criar características de data
    df_route['year'] = df_route.index.year
    df_route['month'] = df_route.index.month
    df_route['week'] = df_route.index.isocalendar().week
    df_route['day'] = df_route.index.dayofweek

    # Variáveis de entrada e saída
    X = df_route[['year', 'month', 'week', 'day']]
    y = df_route['freight_cost']

    # Dividir os dados em treino e teste.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo XGBoost
    model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse'
    )

    # Ajustar o modelo
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )

    # Cria um DataFrame para armazenar previsões futuras.
    future_dates = pd.date_range(start=df_route.index[-1] + pd.Timedelta(weeks=1), periods=n, freq='W')
    future_df = pd.DataFrame({
        'year': future_dates.year,
        'month': future_dates.month,
        'week': future_dates.isocalendar().week,
        'day': future_dates.dayofweek
    }, index=future_dates)

    # Fazer previsões para as próximas 36 semanas
    future_forecast = model.predict(future_df)
    list_forecast.append(pd.DataFrame(index=future_dates,
                                      data={'freight_cost':list(future_forecast),
                                            'id_city_origin': [origin] * len(future_forecast),
                                            'id_city_destination':[destination] * len(future_forecast)}))

    # Plotar os resultados
    plt.figure(figsize=(12, 6))
    plt.plot(df_route.index, df_route['freight_cost'], label='Valor atual')
    plt.plot(future_dates, future_forecast, label='Predição', color='red')
    plt.title('Cotação atual x Cotação prevista')
    plt.savefig(f'{parent_directory}/images/{origin}-{destination}.png')


# Lendo dados salvos na AWS.
df_freight_costs = get_aws_csv(freight_costs_key)
df_distances = get_aws_csv(distances_key)
df_destinations = get_aws_csv(destinations_key)

# Custos dos fretes pelos destinos.
df_freight_costs_distances = get_freight_costs_distances_by_destination(df_freight_costs,
                                                                        df_distances,
                                                                        df_destinations)
destinations = df_destinations['Destination'].values.astype(float).tolist()

# Gera previsões para as próximas n semanas.
list_forecast = []
for origin, destination in product(df_freight_costs_distances['id_city_origin'].unique(), destinations):
    list_forecast.append(generate_forecast(df_freight_costs_distances,
                                           origin,
                                           destination,
                                           52))

# Salva o csv no diretório.
df_forecast = pd.concat(list_forecast)
df_forecast.to_csv(f'{parent_directory}/outputs/Output_2.csv', index_label='dt_reference')