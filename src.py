# Importação de bibliotecas.
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from itertools import product


# Pegando caminho da pasta automaticamente pela execução do arquivo.
parent_directory = os.path.dirname(os.path.abspath(__file__))

# Leitura dos arquivos csv.
df_freight_costs = pd.read_csv(f'{parent_directory}/csv_files/freight_costs.csv',
                               delimiter=';',
                               decimal=',')
df_distances = pd.read_csv(f'{parent_directory}/csv_files/distances.csv',
                               delimiter=';',
                               decimal=',')

# Remoção dos valores nulos.
df_distances.dropna(inplace=True)
df_freight_costs.dropna(inplace=True)

# Unindo as bases de cotações e distâncias.
df_freight_costs_distances = pd.merge(df_freight_costs,
                                      df_distances,
                                      on=['id_city_origin', 'id_city_destination'])

# Lista de destinos especificados
destinations = [1501303, 1506807, 3205309, 3548500, 4118204, 4207304, 4216206, 4315602]

# Filtrando as cotações para os destinos especificados.
df_freight_costs_distances = df_freight_costs_distances[df_freight_costs_distances['id_city_destination'].isin(destinations)]

# Salvando o arquivo csv e gerando o primeiro Output.
df_freight_costs_distances.to_csv(f'{parent_directory}/outputs/Output_1.csv')

# Convertendo as datas de string para TimeStamp da biblioteca Pandas.
df_freight_costs_distances['dt_reference'] = pd.to_datetime(df_freight_costs_distances['dt_reference'], format='%d/%m/%Y')
df_freight_costs_distances.set_index('dt_reference', inplace=True)

# Loop iterando por origem e destino.
list_forecast = []
for origin, destination in product(df_freight_costs_distances['id_city_origin'].unique(), destinations):
        
    # Rota para origem e destino,
    df_route = df_freight_costs_distances[
        (df_freight_costs_distances['id_city_origin'] == origin) & 
        (df_freight_costs_distances['id_city_destination'] == destination)
    ].sort_values(by='dt_reference')

    # Pula, caso a origem e destino tenham menos que 2 dados.
    if len(df_route) < 2:
        continue

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

    # Faz previsões para o conjunto de teste.
    y_pred = model.predict(X_test)

    # Cria um DataFrame para armazenar previsões futuras.
    future_dates = pd.date_range(start=df_route.index[-1] + pd.Timedelta(weeks=1), periods=52, freq='W')
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

# Salva o csv no diretório.
df_forecast = pd.concat(list_forecast)
df_forecast.to_csv(f'{parent_directory}/outputs/Output_2.csv', index_label='dt_reference')