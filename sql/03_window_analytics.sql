-- ============================================================================
-- 03_window_analytics.sql — Advanced Spark SQL Window Functions
-- Techniques: SUM() OVER, AVG() OVER (ROWS BETWEEN), LAG, RANK, NTILE, ROW_NUMBER
-- ============================================================================

-- ── Q1: Cumulative revenue, 7-day rolling trip average & DoD growth (LAG) ───
WITH daily_metrics AS (
    SELECT
        pickup_date,
        COUNT(*)                    AS daily_trips,
        ROUND(SUM(total_amount), 2) AS daily_revenue
    FROM trips
    GROUP BY pickup_date
)
SELECT
    pickup_date,
    daily_trips,
    daily_revenue,
    ROUND(SUM(daily_revenue) OVER (ORDER BY pickup_date), 2) AS cumulative_revenue,
    ROUND(AVG(daily_trips) OVER (
        ORDER BY pickup_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 0)                                                    AS rolling_7d_trips,
    ROUND(AVG(daily_revenue) OVER (
        ORDER BY pickup_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2)                                                    AS rolling_7d_revenue,
    LAG(daily_trips, 1) OVER (ORDER BY pickup_date)          AS prev_day_trips,
    ROUND(
        (daily_trips - LAG(daily_trips, 1) OVER (ORDER BY pickup_date)) * 100.0
        / NULLIF(LAG(daily_trips, 1) OVER (ORDER BY pickup_date), 0),
        2
    )                                                        AS dod_trips_pct_change,
    RANK() OVER (ORDER BY daily_revenue DESC)                AS revenue_rank
FROM daily_metrics
ORDER BY pickup_date;


-- ── Q2: Top 3 Pickup Zones per Borough using ROW_NUMBER() OVER (PARTITION BY) ─
WITH zone_counts AS (
    SELECT
        COALESCE(z.Borough, 'Unknown')          AS borough,
        COALESCE(z.Zone, 'Unknown')             AS zone_name,
        COUNT(*)                                AS trips,
        ROUND(AVG(t.total_amount), 2)           AS avg_fare
    FROM trips t
    LEFT JOIN zones z ON t.pickup_location_id = z.LocationID
    WHERE z.Borough IS NOT NULL AND z.Borough NOT IN ('Unknown', 'N/A')
    GROUP BY COALESCE(z.Borough, 'Unknown'), COALESCE(z.Zone, 'Unknown')
),
ranked_zones AS (
    SELECT
        borough,
        zone_name,
        trips,
        avg_fare,
        ROW_NUMBER() OVER (PARTITION BY borough ORDER BY trips DESC) AS rn,
        ROUND(trips * 100.0 / SUM(trips) OVER (PARTITION BY borough), 2) AS borough_share_pct
    FROM zone_counts
)
SELECT borough, rn AS rank_in_borough, zone_name, trips, borough_share_pct, avg_fare
FROM ranked_zones
WHERE rn <= 3
ORDER BY trips DESC;


-- ── Q3: Hourly demand quartiles using NTILE(4) ──────────────────────────────
WITH hourly AS (
    SELECT
        pickup_hour,
        COUNT(*)                    AS trips,
        ROUND(AVG(total_amount), 2) AS avg_fare
    FROM trips
    GROUP BY pickup_hour
)
SELECT
    pickup_hour,
    trips,
    avg_fare,
    NTILE(4) OVER (ORDER BY trips DESC) AS demand_quartile
FROM hourly
ORDER BY pickup_hour;
