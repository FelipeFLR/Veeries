import boto3

# Constantes da AWS.
bucket_name = 'veeries-challenge'
distances_key = 'csv_files/distances.csv'
freight_costs_key = 'csv_files/freight_costs.csv'
destinations_key = 'csv_files/destinations.csv'

# Instância da AWS.
s3 = boto3.client(
    's3'
)
