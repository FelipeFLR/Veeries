## Análise Exploratória

Antes da implementação do código, foi realizada uma análise exploratória dos dados para compreender as características e padrões presentes. Analisei que os custos de frete apresentam uma sazonalidade significativa ao longo do ano, com valores mais altos em julho e mais baixos em junho.

## Abordagem e Implementação

Com base na análise exploratória, foi escolhido o modelo XGBoost para prever os valores futuros de frete. O código foi desenvolvido de maneira simples, sem a necessidade de implementação de classes. O carregamento das funções são realizadas através da AWS S3, para
carga de arquivos não estruturados.

### Estrutura do Código

1. **Importação de Bibliotecas**
   O código começa com a importação das bibliotecas necessárias:
   ```python
   import pandas as pd
   from xgboost import XGBRegressor
   from sklearn.model_selection import train_test_split
   import matplotlib.pyplot as plt
   import os
   from itertools import product
   ```

2. **Leitura e Preparação dos Dados a partir da leitura dos arquivos na AWS**
3. **Filtragem e Processamento dos Dados**
   Filtragem dos destinos e preparação dos dados para o modelo:
   ```python
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
   ```

4. **Treinamento do Modelo e Previsões**
   Treinamento do modelo XGBoost e realização das previsões:
   ```python
   list_forecast = []
   for origin, destination in product(df_freight_costs_distances['id_city_origin'].unique(), destinations):
       df_route = df_freight_costs_distances[(df_freight_costs_distances['id_city_origin'] == origin) & 
                                             (df_freight_costs_distances['id_city_destination'] == destination)].sort_values(by='dt_reference')
       if len(df_route) < 2:
           continue
       df_route['year'] = df_route.index.year
       df_route['month'] = df_route.index.month
       df_route['week'] = df_route.index.isocalendar().week
       df_route['day'] = df_route.index.dayofweek
       X = df_route[['year', 'month', 'week', 'day']]
       y = df_route['freight_cost']
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model = XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
       model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
       y_pred = model.predict(X_test)
       future_dates = pd.date_range(start=df_route.index[-1] + pd.Timedelta(weeks=1), periods=52, freq='W')
       future_df = pd.DataFrame({'year': future_dates.year, 'month': future_dates.month, 'week': future_dates.isocalendar().week, 'day': future_dates.dayofweek}, index=future_dates)
       future_forecast = model.predict(future_df)
       list_forecast.append(pd.DataFrame(index=future_dates, data={'freight_cost': list(future_forecast), 'id_city_origin': [origin] * len(future_forecast), 'id_city_destination': [destination] * len(future_forecast)}))
       plt.figure(figsize=(12, 6))
       plt.plot(df_route.index, df_route['freight_cost'], label='Valor atual')
       plt.plot(future_dates, future_forecast, label='Predição', color='red')
       plt.title('Cotação atual x Cotação prevista')
       plt.savefig(f'{parent_directory}/images/{origin}-{destination}.png')
   df_forecast = pd.concat(list_forecast)
   df_forecast.to_csv(f'{parent_directory}/outputs/Output_2.csv', index_label='dt_reference')
   ```

### Observações Finais

- **Sazonalidade:** Observou-se sazonalidade significativa nos custos de frete, o que influenciou a escolha do modelo.
- **Código e Melhoria:** O código é simples e direto, mas pode ser melhorado separando-o em funções específicas para modularizar a leitura de dados e a previsão de custos.

## Estrutura dos Diretórios

- `csv_files/`: Contém os arquivos CSV de entrada.
- `outputs/`: Contém os arquivos de saída gerados pelo código.
- `images/`: Contém os gráficos gerados durante a execução do código.

## Requisitos

- `pandas`
- `xgboost`
- `scikit-learn`
- `matplotlib`
- `os`