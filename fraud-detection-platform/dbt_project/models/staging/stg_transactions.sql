{{ config(materialized='view') }}

with source as (
    select * from read_csv_auto('../data/raw/transactions.csv')
),

cleaned as (
    select
        transaction_id,
        customer_id,
        merchant_id,
        amount,
        strptime(timestamp, '%Y-%m-%d %H:%M:%S') as transaction_timestamp,
        payment_method,
        merchant_category,
        is_fraud::boolean as is_fraud,
        extract('hour' from strptime(timestamp, '%Y-%m-%d %H:%M:%S')) as hour_of_day,
        extract('dow' from strptime(timestamp, '%Y-%m-%d %H:%M:%S')) as day_of_week,
        date_trunc('day', strptime(timestamp, '%Y-%m-%d %H:%M:%S')) as transaction_date
    from source
)

select * from cleaned