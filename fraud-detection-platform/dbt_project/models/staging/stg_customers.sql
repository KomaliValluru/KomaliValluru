{{ config(materialized='view') }}

with source as (
    select * from read_csv_auto('../data/raw/customers.csv')
),

cleaned as (
    select
        customer_id,
        round(age::float, 0) as customer_age,
        round(account_age_days::float, 0) as account_age_days,
        round(credit_score::float, 0) as credit_score,
        round(income::float, 2) as annual_income
    from source
    where age between 18 and 100
      and credit_score between 300 and 850
      and income > 0
)

select * from cleaned