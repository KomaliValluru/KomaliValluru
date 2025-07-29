{{ config(materialized='table') }}

with features as (
    select * from {{ ref('int_transaction_features') }}
),

final as (
    select
        transaction_id,
        customer_id,
        merchant_id,
        amount,
        transaction_timestamp,
        transaction_date,
        payment_method,
        merchant_category,
        is_fraud,
        
        -- Customer features
        customer_age,
        account_age_days,
        credit_score,
        annual_income,
        customer_total_transactions,
        customer_avg_amount,
        
        -- Transaction features
        hour_of_day,
        day_of_week,
        amount_zscore,
        is_night_transaction,
        is_weekend_transaction,
        is_amount_outlier,
        
        -- Merchant features
        merchant_risk_score,
        is_high_risk_merchant,
        
        -- Risk score calculation
        (
            case when is_amount_outlier = 1 then 0.3 else 0 end +
            case when is_night_transaction = 1 then 0.2 else 0 end +
            case when is_high_risk_merchant = 1 then 0.25 else 0 end +
            case when abs(amount_zscore) > 2 then 0.25 else 0 end
        ) as calculated_risk_score
        
    from features
)

select * from final