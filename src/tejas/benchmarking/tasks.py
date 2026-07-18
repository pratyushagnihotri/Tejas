"""24 database-specific tasks across three difficulty tiers.

Two task types:
  - "sql": the model must write a SELECT query. Graded by executing both the
    model's query and a trusted "gold" reference query against the same
    seeded SQLite database and comparing result sets (execution accuracy).
  - "concept": a short-answer database concept question, graded by substring
    match against one or more accepted phrases.
"""

from tejas.benchmarking.seed_db import SCHEMA_PREAMBLE

TASKS = [
    # ---------------- easy ----------------
    {"id": "e1", "tier": "easy", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "List the names of all employees in the "
               "'Engineering' department. Output a single column named name.",
     "gold_sql": "SELECT e.name FROM employees e JOIN departments d "
                 "ON e.dept_id = d.dept_id WHERE d.dept_name = 'Engineering'"},
    {"id": "e2", "tier": "easy", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "Count how many departments exist in total. "
               "Output a single column with the count.",
     "gold_sql": "SELECT COUNT(*) FROM departments"},
    {"id": "e3", "tier": "easy", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "Find the name of the employee with the "
               "highest salary. Output a single column named name.",
     "gold_sql": "SELECT name FROM employees ORDER BY salary DESC LIMIT 1"},
    {"id": "e4", "tier": "easy", "type": "concept",
     "prompt": "What does the acronym 'PK' stand for in database terminology? "
               "Reply with just the term.",
     "accept": ["primary key"]},
    {"id": "e5", "tier": "easy", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "List the order_id and amount of all orders "
               "with amount greater than 500.",
     "gold_sql": "SELECT order_id, amount FROM orders WHERE amount > 500"},
    {"id": "e6", "tier": "easy", "type": "concept",
     "prompt": "Which SQL clause is used to filter rows before grouping? "
               "Reply with just the clause name.",
     "accept": ["where"]},
    {"id": "e7", "tier": "easy", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "List the names of all employees hired after "
               "the year 2020. Output a single column named name.",
     "gold_sql": "SELECT name FROM employees WHERE hire_year > 2020"},
    {"id": "e8", "tier": "easy", "type": "concept",
     "prompt": "Which SQL clause is used to filter groups after a GROUP BY? "
               "Reply with just the clause name.",
     "accept": ["having"]},

    # ---------------- medium ----------------
    {"id": "m1", "tier": "medium", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "List each department's name along with the "
               "number of employees in it. Output columns dept_name and "
               "emp_count.",
     "gold_sql": "SELECT d.dept_name, COUNT(e.emp_id) AS emp_count FROM "
                 "departments d LEFT JOIN employees e ON e.dept_id = d.dept_id "
                 "GROUP BY d.dept_name"},
    {"id": "m2", "tier": "medium", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "Find the total order amount handled by each "
               "employee. Output columns name and total_amount, including "
               "only employees who have at least one order.",
     "gold_sql": "SELECT e.name, SUM(o.amount) AS total_amount FROM employees "
                 "e JOIN orders o ON o.emp_id = e.emp_id GROUP BY e.name"},
    {"id": "m3", "tier": "medium", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "List the names of employees who work in a "
               "department with a budget greater than 100000. Output a "
               "single column named name.",
     "gold_sql": "SELECT e.name FROM employees e JOIN departments d ON "
                 "e.dept_id = d.dept_id WHERE d.budget > 100000"},
    {"id": "m4", "tier": "medium", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "Find the average salary per department. "
               "Output columns dept_name and avg_salary.",
     "gold_sql": "SELECT d.dept_name, AVG(e.salary) AS avg_salary FROM "
                 "departments d JOIN employees e ON e.dept_id = d.dept_id "
                 "GROUP BY d.dept_name"},
    {"id": "m5", "tier": "medium", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "List the names of employees who have never "
               "placed an order. Output a single column named name.",
     "gold_sql": "SELECT e.name FROM employees e WHERE e.emp_id NOT IN "
                 "(SELECT emp_id FROM orders)"},
    {"id": "m6", "tier": "medium", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "Find the name of the department with the "
               "highest total employee salary. Output a single column named "
               "dept_name.",
     "gold_sql": "SELECT d.dept_name FROM departments d JOIN employees e ON "
                 "e.dept_id = d.dept_id GROUP BY d.dept_name ORDER BY "
                 "SUM(e.salary) DESC LIMIT 1"},
    {"id": "m7", "tier": "medium", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "List customers who have placed more than 2 "
               "orders, along with their order count. Output columns "
               "customer and order_count.",
     "gold_sql": "SELECT customer, COUNT(*) AS order_count FROM orders "
                 "GROUP BY customer HAVING COUNT(*) > 2"},
    {"id": "m8", "tier": "medium", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "For each employee who has a manager, list "
               "the employee's name and their manager's name. Output columns "
               "emp_name and manager_name.",
     "gold_sql": "SELECT e.name AS emp_name, m.name AS manager_name FROM "
                 "employees e JOIN employees m ON e.manager_id = m.emp_id"},

    # ---------------- hard ----------------
    {"id": "h1", "tier": "hard", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "Find the names of employees whose salary is "
               "above their own department's average salary. Output a "
               "single column named name.",
     "gold_sql": "SELECT e.name FROM employees e WHERE e.salary > "
                 "(SELECT AVG(e2.salary) FROM employees e2 WHERE "
                 "e2.dept_id = e.dept_id)"},
    {"id": "h2", "tier": "hard", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "For each employee, compute their salary "
               "rank within their department (1 = highest paid). Output "
               "columns name, dept_id, and salary_rank.",
     "gold_sql": "SELECT name, dept_id, RANK() OVER (PARTITION BY dept_id "
                 "ORDER BY salary DESC) AS salary_rank FROM employees"},
    {"id": "h3", "tier": "hard", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "Find the names of the top 3 employees by "
               "total order amount. Output a single column named name.",
     "gold_sql": "SELECT e.name FROM employees e JOIN orders o ON "
                 "o.emp_id = e.emp_id GROUP BY e.name ORDER BY "
                 "SUM(o.amount) DESC LIMIT 3"},
    {"id": "h4", "tier": "hard", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "Find the names of departments where the "
               "combined order amount of all their employees exceeds 5000. "
               "Output a single column named dept_name.",
     "gold_sql": "SELECT d.dept_name FROM departments d JOIN employees e ON "
                 "e.dept_id = d.dept_id JOIN orders o ON o.emp_id = e.emp_id "
                 "GROUP BY d.dept_name HAVING SUM(o.amount) > 5000"},
    {"id": "h5", "tier": "hard", "type": "concept",
     "prompt": "In query optimization, what is the term for the estimated "
               "number of rows a query operator will produce? Reply with "
               "just the term.",
     "accept": ["cardinality"]},
    {"id": "h6", "tier": "hard", "type": "concept",
     "prompt": "Which SQL transaction isolation level prevents dirty reads "
               "but still allows non-repeatable reads? Reply with just the "
               "isolation level name.",
     "accept": ["read committed"]},
    {"id": "h7", "tier": "hard", "type": "concept",
     "prompt": "Which normal form eliminates transitive functional "
               "dependencies on the primary key? Reply with just the normal "
               "form.",
     "accept": ["3nf", "third normal form"]},
    {"id": "h8", "tier": "hard", "type": "sql",
     "prompt": SCHEMA_PREAMBLE + "For each department, find the name of the "
               "employee with the highest salary in that department. Output "
               "columns dept_name and name.",
     "gold_sql": "SELECT d.dept_name, e.name FROM departments d JOIN "
                 "employees e ON e.dept_id = d.dept_id WHERE e.salary = "
                 "(SELECT MAX(e2.salary) FROM employees e2 WHERE "
                 "e2.dept_id = e.dept_id)"},
]
