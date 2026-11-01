-- ============================================================================
-- 04_ab_experiment_cohorts.sql — Vendor A/B Cohorts & Confounder Stratification
-- Group A (Control)   : Vendor 1 = Creative Mobile Technologies (CMT)
-- Group B (Treatment) : Vendor 2 = VeriFone
-- ============================================================================

-- ── Q1: Overall Vendor A/B Summary & Telemetry Completeness ─────────────────
SELECT
    vendor_id,
    CASE vendor_id WHEN 1 THEN 'CMT (Vendor 1)' ELSE 'VeriFone (Vendor 2)' END AS vendor_name,
    COUNT(*)                                                    AS trips,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)          AS share_pct,
    ROUND(AVG(total_amount), 2)                                 AS mean_total_fare,
    ROUND(PERCENTILE_APPROX(total_amount, 0.5), 2)              AS median_total_fare,
    ROUND(STDDEV(total_amount), 2)                              AS std_total_fare,
    ROUND(AVG(trip_distance), 2)                                AS mean_distance_mi,
    ROUND(AVG(trip_minutes), 2)                                 AS mean_duration_min,
    ROUND(SUM(CASE WHEN is_airport_trip = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
                                                                AS airport_trip_pct,
    ROUND(SUM(CASE WHEN passenger_count IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
                                                                AS null_telemetry_pct
FROM trips
WHERE vendor_id IN (1, 2)
GROUP BY vendor_id
ORDER BY vendor_id;


-- ── Q2: Credit-Card Tip Behavior & Generous Tipper Rate (≥20%) by Vendor ────
SELECT
    vendor_id,
    CASE vendor_id WHEN 1 THEN 'CMT (Vendor 1)' ELSE 'VeriFone (Vendor 2)' END AS vendor_name,
    COUNT(*)                                                    AS cc_trips,
    ROUND(AVG(tip_amount), 2)                                   AS mean_tip_usd,
    ROUND(AVG(tip_pct), 2)                                      AS mean_tip_pct,
    ROUND(PERCENTILE_APPROX(tip_pct, 0.5), 2)                   AS median_tip_pct,
    ROUND(SUM(CASE WHEN is_generous_tip = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
                                                                AS generous_tip_rate_pct,
    ROUND(SUM(CASE WHEN tip_amount = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
                                                                AS zero_tip_rate_pct
FROM trips
WHERE vendor_id IN (1, 2)
  AND payment_type = 1
  AND fare_amount >= 3.0
GROUP BY vendor_id
ORDER BY vendor_id;


-- ── Q3: Confounder Stratification — Vendor Fare Gap Within Route/Distance Strata ─
-- Shows how route composition (Airport vs City + Distance Bucket) confounds naive A/B means
SELECT
    CASE WHEN is_airport_trip = 1 THEN 'Airport Trip' ELSE 'Standard City Trip' END AS trip_type,
    distance_bucket,
    SUM(CASE WHEN vendor_id = 1 THEN 1 ELSE 0 END)              AS cmt_trips,
    SUM(CASE WHEN vendor_id = 2 THEN 1 ELSE 0 END)              AS verifone_trips,
    ROUND(AVG(CASE WHEN vendor_id = 1 THEN total_amount END), 2) AS cmt_mean_fare,
    ROUND(AVG(CASE WHEN vendor_id = 2 THEN total_amount END), 2) AS verifone_mean_fare,
    ROUND(
        AVG(CASE WHEN vendor_id = 2 THEN total_amount END) -
        AVG(CASE WHEN vendor_id = 1 THEN total_amount END),
        2
    )                                                           AS raw_stratum_diff
FROM trips
WHERE vendor_id IN (1, 2)
GROUP BY
    CASE WHEN is_airport_trip = 1 THEN 'Airport Trip' ELSE 'Standard City Trip' END,
    distance_bucket
ORDER BY
    trip_type DESC,
    CASE distance_bucket
        WHEN '< 1 mi'  THEN 1
        WHEN '1-3 mi'  THEN 2
        WHEN '3-5 mi'  THEN 3
        WHEN '5-10 mi' THEN 4
        ELSE 5
    END;


-- ── Q4: Subgroup Heterogeneity Across Time-of-Day Buckets ───────────────────
SELECT
    time_bucket,
    vendor_id,
    COUNT(*)                                                    AS trips,
    ROUND(AVG(total_amount), 2)                                 AS mean_fare,
    ROUND(STDDEV(total_amount), 2)                              AS std_fare,
    ROUND(AVG(trip_distance), 2)                                AS mean_distance
FROM trips
WHERE vendor_id IN (1, 2)
GROUP BY time_bucket, vendor_id
ORDER BY time_bucket, vendor_id;
