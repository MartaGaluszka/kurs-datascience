-- 1. Dodajemy przykładowy produkt do tabeli
INSERT INTO products (name, category, price, stock, description)
VALUES ('MacBook Air M3', 'Laptopy', 5999.00, 15, 'Lekki i potężny laptop do nauki Data Science');

-- 2. Odświeżamy widok, żeby zobaczyć wszystkie produkty
SELECT * FROM products;