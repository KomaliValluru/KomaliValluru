{{ config(materialized='view') }}

with transactions as (
    select * from {{ ref('stg_transactions') }}
),

customers as (
    select * from {{ ref('stg_customers') }}
),

merchants as (
    select * from {{ ref('stg_merchants') }}
),

transaction_stats as (
    select
        customer_id,
        count(*) as total_transactions,
        avg(amount) as avg_transaction_amount,
        stddev(amount) as stddev_transaction_amount,
        max(amount) as max_transaction_amount,
        min(amount) as min_transaction_amount
    from transactions
    group by customer_id
),

hourly_patterns as (
    select
        customer_id,
        hour_of_day,
        count(*) as transactions_in_hour,
        avg(amount) as avg_amount_in_hour
    from transactions
    group by customer_id, hour_of_day
),

enriched_transactions as (
    select
        t.*,
        c.customer_age,
        c.account_age_days,
        c.credit_score,
        c.annual_income,
        m.merchant_risk_score,
        m.merchant_location,
        ts.total_transactions as customer_total_transactions,
        ts.avg_transaction_amount as customer_avg_amount,
        ts.stddev_transaction_amount as customer_stddev_amount,
        
        -- Transaction amount percentiles for the customer
        case 
            when ts.stddev_transaction_amount = 0 then 0.5
            else (t.amount - ts.avg_transaction_amount) / ts.stddev_transaction_amount
        end as amount_zscore,
        
        -- Time-based features
        case when t.hour_of_day between 22 and 6 then 1 else 0 end as is_night_transaction,
        case when t.day_of_week in (0, 6) then 1 else 0 end as is_weekend_transaction,
        
        -- Risk indicators
        case when t.amount > ts.avg_transaction_amount + 2 * ts.stddev_transaction_amount then 1 else 0 end as is_amount_outlier,
        case when m.merchant_risk_score > 0.7 then 1 else 0 end as is_high_risk_merchant
        
    from transactions t
    left join customers c on t.customer_id = c.customer_id
    left join merchants m on t.merchant_id = m.merchant_id
    left join transaction_stats ts on t.customer_id = ts.customer_id
)

select * from enriched_transactions