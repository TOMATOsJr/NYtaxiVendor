-- ============================================================================
-- 02_demand_and_velocity.sql — Demand Matrix, Traffic Speed & Trip Profiles
-- Dataset: NYC TLC Yellow Taxi (Jan 2024)
-- ============================================================================

-- ── Q1: Hour × Day-of-Week demand & average traffic speed matrix ────────────
SELECT
    pickup_hour,
    pickup_dow,
    COUNT(*)                                    AS trips,
    ROUND(AVG(speed_mph), 2)                    AS avg_speed_mph,
    ROUND(AVG(total_amount), 2)                 AS avg_fare
FROM trips
WHERE speed_mph BETWEEN 1.0 AND 65.0
GROUP BY pickup_hour, pickup_dow
ORDER BY pickup_hour, pickup_dow;


-- ── Q2: Hourly velocity profile & congestion index ──────────────────────────
SELECT
    pickup_hour,
    COUNT(*)                                    AS trips,
    ROUND(AVG(speed_mph), 2)                    AS avg_speed_mph,
    ROUND(PERCENTILE_APPROX(speed_mph, 0.5), 2) AS median_speed_mph,
    ROUND(AVG(trip_minutes), 2)                 AS avg_duration_min
FROM trips
WHERE speed_mph BETWEEN 1.0 AND 65.0
GROUP BY pickup_hour
ORDER BY pickup_hour;


-- ── Q3: Distance bucket profile & revenue contribution ──────────────────────
SELECT
    distance_bucket,
    COUNT(*)                                    AS trips,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_trips,
    ROUND(AVG(total_amount), 2)                 AS avg_fare,
    ROUND(AVG(trip_minutes), 2)                 AS avg_duration_min,
    ROUND(AVG(speed_mph), 2)                    AS avg_speed_mph
FROM trips
GROUP BY distance_bucket
ORDER BY
    CASE distance_bucket
        WHEN '< 1 mi'  THEN 1
        WHEN '1-3 mi'  THEN 2
        WHEN '3-5 mi'  THEN 3
        WHEN '5-10 mi' THEN 4
        ELSE 5
    END;


-- ── Q4: Top 12 Origin-Destination Zone Corridors ────────────────────────────
SELECT
    COALESCE(zp.Zone, CAST(t.pickup_location_id AS STRING))  AS pickup_zone,
    COALESCE(zd.Zone, CAST(t.dropoff_location_id AS STRING)) AS dropoff_zone,
    COUNT(*)                                                 AS trips,
    ROUND(AVG(t.total_amount), 2)                            AS avg_fare,
    ROUND(AVG(t.trip_minutes), 1)                            AS avg_minutes
FROM trips t
LEFT JOIN zones zp ON t.pickup_location_id  = zp.LocationID
LEFT JOIN zones zd ON t.dropoff_location_id = zd.LocationID
WHERE t.pickup_location_id != t.dropoff_location_id
GROUP BY
    COALESCE(zp.Zone, CAST(t.pickup_location_id AS STRING)),
    COALESCE(zd.Zone, CAST(t.dropoff_location_id AS STRING))
ORDER BY trips DESC
LIMIT 12;
