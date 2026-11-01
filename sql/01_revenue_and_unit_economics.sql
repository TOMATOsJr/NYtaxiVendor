-- ============================================================================
-- 01_revenue_and_unit_economics.sql — Spark SQL Revenue & Unit Economics
-- Dataset: NYC TLC Yellow Taxi (Jan 2024) + Official Taxi Zone Lookup
-- Views:   trips (cleaned Spark table), zones (borough & zone lookup)
-- ============================================================================

-- ── Q1: Daily revenue, trip volume, and average fare ────────────────────────
SELECT
    pickup_date,
    COUNT(*)                                    AS trips,
    ROUND(SUM(total_amount), 2)                 AS daily_revenue,
    ROUND(AVG(total_amount), 2)                 AS avg_total_fare,
    ROUND(AVG(fare_amount), 2)                  AS avg_base_fare,
    ROUND(AVG(tip_amount), 2)                   AS avg_tip
FROM trips
GROUP BY pickup_date
ORDER BY pickup_date;


-- ── Q2: Unit-economics fee decomposition across January 2024 ────────────────
SELECT
    COUNT(*)                                                AS total_trips,
    ROUND(SUM(fare_amount) / 1e6, 2)                        AS base_fare_m,
    ROUND(SUM(tip_amount) / 1e6, 2)                         AS tips_m,
    ROUND(SUM(tolls_amount) / 1e6, 2)                       AS tolls_m,
    ROUND(SUM(COALESCE(congestion_surcharge, 0)) / 1e6, 2)  AS congestion_m,
    ROUND(SUM(COALESCE(airport_fee, 0)) / 1e6, 2)           AS airport_fee_m,
    ROUND(SUM(total_amount) / 1e6, 2)                       AS total_revenue_m,
    ROUND(SUM(fare_amount) * 100.0 / SUM(total_amount), 1)  AS base_share_pct,
    ROUND(SUM(tip_amount) * 100.0 / SUM(total_amount), 1)   AS tip_share_pct
FROM trips;


-- ── Q3: Revenue & fare premium by Pickup Borough (Spark SQL Join) ───────────
SELECT
    COALESCE(z.Borough, 'Unknown')              AS pickup_borough,
    COUNT(*)                                    AS trips,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS trip_share_pct,
    ROUND(SUM(t.total_amount) / 1e6, 2)         AS revenue_m,
    ROUND(AVG(t.total_amount), 2)               AS avg_fare,
    ROUND(AVG(t.trip_distance), 2)              AS avg_distance_mi,
    ROUND(AVG(t.trip_minutes), 2)               AS avg_duration_min
FROM trips t
LEFT JOIN zones z
    ON t.pickup_location_id = z.LocationID
GROUP BY COALESCE(z.Borough, 'Unknown')
ORDER BY trips DESC;


-- ── Q4: Top 15 Pickup Zones with human-readable Zone & Borough names ────────
SELECT
    t.pickup_location_id                        AS zone_id,
    COALESCE(z.Zone, 'Unknown Zone')            AS zone_name,
    COALESCE(z.Borough, 'Unknown')              AS borough,
    COUNT(*)                                    AS trips,
    ROUND(AVG(t.total_amount), 2)               AS avg_fare,
    ROUND(AVG(t.trip_distance), 2)              AS avg_distance_mi
FROM trips t
LEFT JOIN zones z
    ON t.pickup_location_id = z.LocationID
GROUP BY t.pickup_location_id, COALESCE(z.Zone, 'Unknown Zone'), COALESCE(z.Borough, 'Unknown')
ORDER BY trips DESC
LIMIT 15;
