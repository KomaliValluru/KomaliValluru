{{ config(materialized='view') }}

with source as (
    select * from read_csv_auto('../data/raw/merchants.csv')
),

cleaned as (
    select
        merchant_id,
        merchant_name,
        category as merchant_category,
        round(risk_score::float, 3) as merchant_risk_score,
        location as merchant_location
    from source
)

select * from cleaned