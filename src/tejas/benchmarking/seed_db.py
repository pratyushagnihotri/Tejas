"""Creates a small, deterministic SQLite database the routing-spike SQL tasks
run against. Re-running always produces the exact same DB (existing file is
replaced), so gold and candidate queries are graded against identical data.
"""

import os
import sqlite3

from tejas.paths import COMPANY_DB

DB_PATH = str(COMPANY_DB)

DEPARTMENTS = [
    # dept_id, dept_name, budget
    (1, "Engineering", 250000),
    (2, "Sales", 150000),
    (3, "Marketing", 90000),
    (4, "HR", 60000),
    (5, "Finance", 120000),
]

# emp_id, name, dept_id, salary, hire_year, manager_id
EMPLOYEES = [
    (1, "Alice Chen", 1, 95000, 2018, None),
    (2, "Brian Kim", 1, 88000, 2019, 1),
    (3, "Carla Diaz", 1, 91000, 2017, 1),
    (4, "David Lee", 1, 76000, 2021, 2),
    (5, "Elena Petrov", 1, 82000, 2020, 2),
    (6, "Frank Osei", 2, 72000, 2019, None),
    (7, "Grace Lin", 2, 68000, 2020, 6),
    (8, "Hassan Ali", 2, 71000, 2018, 6),
    (9, "Irene Novak", 2, 65000, 2022, 7),
    (10, "Jack Turner", 3, 60000, 2019, None),
    (11, "Karen Wu", 3, 58000, 2021, 10),
    (12, "Liam O'Brien", 3, 62000, 2020, 10),
    (13, "Maria Santos", 4, 55000, 2018, None),
    (14, "Noah Kim", 4, 53000, 2021, 13),
    (15, "Olivia Brooks", 5, 84000, 2017, None),
    (16, "Peter Zhang", 5, 79000, 2019, 15),
    (17, "Quinn Adams", 5, 77000, 2020, 15),
    (18, "Rosa Fernandez", 1, 99000, 2015, None),
    (19, "Sam Okafor", 2, 74000, 2021, 6),
    (20, "Tina Brandt", 3, 61000, 2022, 10),
]

# Employees who take orders (the other 6 — 1, 6, 10, 13, 15, 18 — never place one,
# which is what task m5 tests for).
ORDER_TAKERS = [2, 3, 4, 5, 7, 8, 9, 11, 12, 14, 16, 17, 19, 20]
CUSTOMERS = ["Acme Corp", "Globex", "Initech", "Umbrella Inc",
             "Wayne Ent", "Stark Ind", "Wonka Co", "Hooli"]


def build_orders():
    orders = []
    oid = 1
    for i, emp_id in enumerate(ORDER_TAKERS):
        for k in range(2):
            customer = CUSTOMERS[(i * 2 + k) % len(CUSTOMERS)]
            amount = 150 + ((i * 37 + k * 91) % 12) * 60
            month = 1 + ((i + k) % 3)
            day = 3 + ((i * 5 + k * 11) % 25)
            date = f"2023-{month:02d}-{day:02d}"
            orders.append((oid, emp_id, customer, float(amount), date))
            oid += 1
    return orders


def create_db(path=DB_PATH):
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE departments (
            dept_id INTEGER PRIMARY KEY,
            dept_name TEXT NOT NULL,
            budget INTEGER NOT NULL
        );
        CREATE TABLE employees (
            emp_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dept_id INTEGER REFERENCES departments(dept_id),
            salary INTEGER NOT NULL,
            hire_year INTEGER NOT NULL,
            manager_id INTEGER REFERENCES employees(emp_id)
        );
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            emp_id INTEGER REFERENCES employees(emp_id),
            customer TEXT NOT NULL,
            amount REAL NOT NULL,
            order_date TEXT NOT NULL
        );
        """
    )
    conn.executemany("INSERT INTO departments VALUES (?,?,?)", DEPARTMENTS)
    conn.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?)", EMPLOYEES)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", build_orders())
    conn.commit()
    conn.close()
    return path


SCHEMA_PREAMBLE = """You are given a SQLite database with this schema:

departments(dept_id INTEGER PRIMARY KEY, dept_name TEXT, budget INTEGER)
employees(emp_id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER REFERENCES departments, salary INTEGER, hire_year INTEGER, manager_id INTEGER REFERENCES employees)
orders(order_id INTEGER PRIMARY KEY, emp_id INTEGER REFERENCES employees, customer TEXT, amount REAL, order_date TEXT)

Write a single SQLite SELECT query for the following task. Reply with ONLY the raw SQL — no explanation, no markdown code fences.

Task: """

if __name__ == "__main__":
    p = create_db()
    print(f"Seeded {p}")
