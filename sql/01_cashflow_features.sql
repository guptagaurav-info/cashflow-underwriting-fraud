-- Cash-flow features from CLASSIFIED transactions (category = classifier/agent output, not ground truth).
-- Naming: {category}_{flow}_{stat}_{window}. Windows are relative to the 6 submitted statement months:
-- 6m = all months, 3m = the 3 most recent months before the application.
CREATE OR REPLACE TABLE txn AS
SELECT t.app_id, t.date, t.amount, t.balance, t.description, t.category_pred AS cat,
       DATE_TRUNC('month', t.date) AS month,
       DATE_DIFF('month', DATE_TRUNC('month', t.date), DATE_TRUNC('month', a.app_date)) AS months_before_app
FROM read_parquet(getvariable('txn_path')) t
JOIN read_parquet(getvariable('app_path')) a ON a.app_id = t.app_id AND a.is_primary;

CREATE OR REPLACE TABLE monthly AS
SELECT app_id, month, ANY_VALUE(months_before_app) AS mba,
       SUM(amount) FILTER (WHERE cat = 'revenue')                         AS revenue,
       SUM(amount) FILTER (WHERE cat = 'transfer_in')                     AS transfer_in,
       SUM(amount) FILTER (WHERE cat = 'loan_proceeds')                   AS loan_proceeds,
       -SUM(amount) FILTER (WHERE cat = 'debt_mca')                       AS mca_out,
       -SUM(amount) FILTER (WHERE cat = 'debt_loan')                      AS loan_out,
       -SUM(amount) FILTER (WHERE cat IN ('payroll','rent','cogs_opex','utilities')) AS opex_out,
       -SUM(amount) FILTER (WHERE cat = 'payroll')                        AS payroll_out,
       -SUM(amount) FILTER (WHERE cat = 'tax')                            AS tax_out,
       -SUM(amount) FILTER (WHERE cat = 'owner_draw')                     AS draw_out,
       COUNT(*)     FILTER (WHERE cat = 'nsf_fee')                        AS nsf_count,
       -SUM(amount) FILTER (WHERE amount < 0)                             AS total_out
FROM txn GROUP BY 1, 2;

CREATE OR REPLACE TABLE daily_bal AS            -- end-of-day balance from the running-balance column
SELECT app_id, date, LAST(balance ORDER BY rowid) AS eod_balance
FROM (SELECT *, ROW_NUMBER() OVER () AS rowid FROM txn) GROUP BY 1, 2;

CREATE OR REPLACE TABLE cf_features AS
WITH m AS (
    SELECT app_id,
           AVG(COALESCE(revenue, 0))                                         AS revenue_in_avg_6m,
           AVG(COALESCE(revenue, 0)) FILTER (WHERE mba <= 3)                 AS revenue_in_avg_3m,
           AVG(COALESCE(revenue, 0)) FILTER (WHERE mba > 3)                  AS revenue_in_avg_prior3m,
           STDDEV_SAMP(COALESCE(revenue, 0)) / NULLIF(AVG(COALESCE(revenue, 0)), 0) AS revenue_cv_6m,
           MIN(COALESCE(revenue, 0)) / NULLIF(AVG(COALESCE(revenue, 0)), 0)  AS revenue_min_to_avg_6m,
           SUM(COALESCE(transfer_in, 0)) / NULLIF(SUM(COALESCE(revenue, 0) + COALESCE(transfer_in, 0)), 0) AS transfer_in_share_6m,
           SUM(COALESCE(loan_proceeds, 0)) FILTER (WHERE mba <= 3)           AS loanproceeds_in_sum_3m,
           SUM(COALESCE(mca_out, 0)) FILTER (WHERE mba <= 3) / NULLIF(SUM(COALESCE(revenue, 0)) FILTER (WHERE mba <= 3), 0) AS mca_out_to_revenue_3m,
           SUM(COALESCE(mca_out, 0) + COALESCE(loan_out, 0)) / NULLIF(SUM(COALESCE(revenue, 0)), 0) AS debt_service_to_revenue_6m,
           SUM(COALESCE(opex_out, 0)) / NULLIF(SUM(COALESCE(revenue, 0)), 0) AS opex_to_revenue_6m,
           SUM(COALESCE(payroll_out, 0)) / NULLIF(SUM(COALESCE(revenue, 0)), 0) AS payroll_to_revenue_6m,
           SUM(COALESCE(draw_out, 0)) / NULLIF(SUM(COALESCE(revenue, 0)), 0) AS owner_draw_to_revenue_6m,
           SUM(nsf_count)                                                    AS nsf_count_6m,
           SUM(nsf_count) FILTER (WHERE mba <= 3)                            AS nsf_count_3m,
           AVG(COALESCE(mca_out, 0) + COALESCE(loan_out, 0))                 AS debt_service_avg_6m,
           AVG(COALESCE(revenue, 0) - COALESCE(opex_out, 0) - COALESCE(tax_out, 0) - COALESCE(draw_out, 0)) AS operating_cf_avg_6m,
           AVG(COALESCE(total_out, 0))                                       AS outflow_avg_6m
    FROM monthly GROUP BY 1
), b AS (
    SELECT app_id, MIN(eod_balance) AS balance_min_6m, AVG(eod_balance) AS balance_avg_6m,
           LAST(eod_balance ORDER BY date) AS balance_end,
           COUNT(*) FILTER (WHERE eod_balance < 0) AS days_negative_6m,
           COUNT(*) FILTER (WHERE eod_balance < 0 AND date >= (SELECT MAX(date) FROM daily_bal d2 WHERE d2.app_id = daily_bal.app_id) - INTERVAL 90 DAY) AS days_negative_3m
    FROM daily_bal GROUP BY 1
), f AS (
    SELECT app_id,
           COUNT(DISTINCT REGEXP_EXTRACT(description, '(FORA FINANCIAL|YELLOWSTONE CAP|PEARL CAPITAL|LIBERTAS FUNDING|KAPITUS|ONDECK CAPITAL|RAPID FINANCE|CREDIBLY|BRISK CAPITAL|EVEREST BUS FUNDING|MULLIGAN FUNDING|IOU CENTRAL)'))
               FILTER (WHERE cat = 'debt_mca' AND months_before_app <= 3) AS mca_out_distinct_funders_3m,
           COUNT(DISTINCT date) FILTER (WHERE cat = 'revenue') * 1.0 / NULLIF(COUNT(DISTINCT date), 0) AS revenue_in_day_share_6m,
           MAX(amount) FILTER (WHERE cat = 'revenue') / NULLIF(SUM(amount) FILTER (WHERE cat = 'revenue'), 0) AS revenue_in_max_share_6m
    FROM txn GROUP BY 1
)
SELECT a.app_id, a.app_date, a.industry, a.years_in_business, a.bureau_score, a.requested_amount,
       m.*, b.balance_min_6m, b.balance_avg_6m, b.balance_end, b.days_negative_6m, b.days_negative_3m,
       f.mca_out_distinct_funders_3m, f.revenue_in_day_share_6m, f.revenue_in_max_share_6m,
       m.revenue_in_avg_3m / NULLIF(m.revenue_in_avg_prior3m, 0)                      AS revenue_trend_3v3,
       b.balance_avg_6m / NULLIF(m.revenue_in_avg_6m, 0)                               AS balance_avg_to_revenue_6m,
       -- liquidity runway: days of average outflows the average balance would cover
       b.balance_avg_6m / NULLIF(m.outflow_avg_6m / 30.0, 0)                           AS liquidity_runway_days,
       -- DSCR on existing debt, and pro-forma with the requested loan
       LEAST(m.operating_cf_avg_6m / NULLIF(m.debt_service_avg_6m, 0), 20)             AS dscr_existing_6m,
       m.operating_cf_avg_6m / NULLIF(m.debt_service_avg_6m + a.requested_amount * getvariable('factor') / getvariable('term'), 0) AS dscr_proforma,
       a.requested_amount / NULLIF(m.revenue_in_avg_6m, 0)                             AS ask_to_monthly_revenue
FROM read_parquet(getvariable('app_path')) a
JOIN m USING (app_id) LEFT JOIN b USING (app_id) LEFT JOIN f USING (app_id)
WHERE a.is_primary;
