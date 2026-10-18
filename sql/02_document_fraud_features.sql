-- Document-integrity features (the "is this statement real?" layer), computed on the statement
-- AS SUBMITTED. None of these use the classifier, so they are robust to categorisation errors.
CREATE OR REPLACE TABLE doc_features AS
WITH t AS (
    SELECT app_id, date, amount, balance, description, seq,
           LAG(balance) OVER (PARTITION BY app_id ORDER BY seq) AS prev_balance
    FROM read_parquet(getvariable('txn_path'))
), recon_rows AS (     -- does each running balance equal the previous balance + this transaction?
    SELECT app_id,
           AVG(CASE WHEN prev_balance IS NOT NULL AND ABS(prev_balance + amount - balance) > 0.02 THEN 1 ELSE 0 END) AS running_balance_break_share
    FROM t GROUP BY 1
), recon_hdr AS (       -- do the statement headers tie out? opening + credits - debits = closing
    SELECT app_id,
           MAX(ABS(opening_balance + total_credits - total_debits - closing_balance) / NULLIF(total_credits, 0)) AS header_recon_gap_max,
           ANY_VALUE(pdf_producer) AS pdf_producer,
           MAX(created_modified_gap_days) AS created_modified_gap_days
    FROM read_parquet(getvariable('stmt_path')) GROUP BY 1
), amt AS (
    SELECT app_id,
           SUM(amount) FILTER (WHERE amount > 0 AND amount >= 1000 AND amount % 1000 = 0) / NULLIF(SUM(amount) FILTER (WHERE amount > 0), 0) AS round_thousand_credit_share,
           SUM(amount) FILTER (WHERE amount > 0 AND REGEXP_MATCHES(description, 'MOBILE DEPOSIT|COUNTER DEPOSIT')) / NULLIF(SUM(amount) FILTER (WHERE amount > 0), 0) AS mobile_deposit_share,
           COUNT(*) AS n_txn
    FROM t GROUP BY 1
), benford AS (         -- mean absolute deviation of first digits from Benford's law
    WITH d AS (SELECT app_id, CAST(LEFT(CAST(CAST(FLOOR(ABS(amount)) AS BIGINT) AS VARCHAR), 1) AS INT) AS digit
               FROM t WHERE ABS(amount) >= 10),
         c AS (SELECT app_id, digit, COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY app_id) AS share FROM d GROUP BY 1, 2)
    SELECT app_id, AVG(ABS(share - LOG10(1 + 1.0 / digit))) AS benford_mad FROM c GROUP BY 1
), dup AS (             -- share of non-recurring-type amounts that repeat to the cent across >= 3 months
    WITH x AS (SELECT app_id, amount, COUNT(DISTINCT DATE_TRUNC('month', date)) AS n_months
               FROM t WHERE NOT REGEXP_MATCHES(description, 'RENT|LEASE|LOAN|FUNDING|ACH DEBIT|DAILY PMT|NET PAY|PAYROLL|FEE')
               GROUP BY 1, 2)
    SELECT app_id, SUM(CASE WHEN n_months >= 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS repeated_amount_share FROM x GROUP BY 1
)
SELECT r.app_id, r.running_balance_break_share, h.header_recon_gap_max, h.pdf_producer,
       CASE WHEN h.pdf_producer IN ('Adobe Acrobat Pro 2024','iLovePDF','Sejda PDF Editor','PDFescape','Smallpdf','ReportLab PDF Library')
            THEN 1 ELSE 0 END AS producer_is_editor,
       h.created_modified_gap_days, a.round_thousand_credit_share, a.mobile_deposit_share, a.n_txn,
       bf.benford_mad, d.repeated_amount_share
FROM recon_rows r JOIN recon_hdr h USING (app_id) JOIN amt a USING (app_id)
LEFT JOIN benford bf USING (app_id) LEFT JOIN dup d USING (app_id);
