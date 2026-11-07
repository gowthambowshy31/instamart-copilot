-- Seven business questions on data/instamart.db. Only completed orders count as revenue or spend.
-- Each query starts with a line "-- name: Qn"; src/part_a_analysis.py runs them one by one.

-- name: Q1
-- WHERE then HAVING: customers whose completed orders above INR 500 add up to more than INR 8000.
SELECT customer_id,
       COUNT(*)                     AS big_orders,
       ROUND(SUM(order_amount), 2)  AS big_order_total
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY big_order_total DESC;

-- name: Q2
-- Three-table join: completed revenue by city and product category.
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.product_id  = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- name: Q3
-- Scalar subquery: products priced above the average catalogue price.
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- name: Q4
-- LEFT JOIN: customers who have no completed order.
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- name: Q5
-- Two CTEs: customers whose total completed spend is above the average customer total.
WITH totals AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average AS (
    SELECT AVG(total_spend) AS avg_spend FROM totals
)
SELECT t.customer_id,
       ROUND(t.total_spend, 2) AS total_spend,
       ROUND(a.avg_spend, 2)   AS avg_spend
FROM totals t CROSS JOIN average a
WHERE t.total_spend > a.avg_spend
ORDER BY t.total_spend DESC;

-- name: Q6
-- Ranking window functions: customers ranked by completed spend inside their city.
WITH totals AS (
    SELECT o.customer_id, c.city, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
)
SELECT city, customer_id, total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM totals
ORDER BY city, row_num;

-- name: Q7
-- Running total: daily completed revenue and the cumulative revenue up to that day.
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY day
)
SELECT day,
       ROUND(revenue, 2)                               AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2)      AS cumulative_revenue
FROM daily
ORDER BY day;
