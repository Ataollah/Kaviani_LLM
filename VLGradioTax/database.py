import sqlite3
from datetime import datetime

DATABASE_PATH = "tax_system.db"

def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS provinces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            code TEXT UNIQUE NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            province_id INTEGER,
            FOREIGN KEY (province_id) REFERENCES provinces(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            tax_rate REAL NOT NULL,
            description TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS citizens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            national_code TEXT UNIQUE NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            father_name TEXT,
            birth_date TEXT,
            city_id INTEGER,
            job_category_id INTEGER,
            annual_income REAL DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (city_id) REFERENCES cities(id),
            FOREIGN KEY (job_category_id) REFERENCES job_categories(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tax_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            citizen_id INTEGER NOT NULL,
            year INTEGER NOT NULL,
            income REAL NOT NULL,
            tax_amount REAL NOT NULL,
            is_paid INTEGER DEFAULT 0,
            payment_date TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (citizen_id) REFERENCES citizens(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tax_exemptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            percentage REAL NOT NULL,
            description TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS citizen_exemptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            citizen_id INTEGER NOT NULL,
            exemption_id INTEGER NOT NULL,
            valid_from TEXT,
            valid_until TEXT,
            FOREIGN KEY (citizen_id) REFERENCES citizens(id),
            FOREIGN KEY (exemption_id) REFERENCES tax_exemptions(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def seed_database():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM provinces")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return
    
    provinces = [
        ("تهران", "01"),
        ("اصفهان", "02"),
        ("فارس", "03"),
        ("خراسان رضوی", "04"),
        ("آذربایجان شرقی", "05"),
        ("مازندران", "06"),
        ("گیلان", "07"),
        ("کرمان", "08"),
    ]
    cursor.executemany("INSERT INTO provinces (name, code) VALUES (?, ?)", provinces)
    
    cities = [
        ("تهران", 1),
        ("کرج", 1),
        ("اصفهان", 2),
        ("کاشان", 2),
        ("شیراز", 3),
        ("مشهد", 4),
        ("تبریز", 5),
        ("ساری", 6),
        ("رشت", 7),
        ("کرمان", 8),
    ]
    cursor.executemany("INSERT INTO cities (name, province_id) VALUES (?, ?)", cities)
    
    job_categories = [
        ("کارمند دولتی", 0.10, "کارکنان بخش دولتی"),
        ("کارمند خصوصی", 0.12, "کارکنان بخش خصوصی"),
        ("آزاد", 0.15, "مشاغل آزاد"),
        ("پزشک", 0.20, "پزشکان و کادر درمان"),
        ("وکیل", 0.18, "وکلا و مشاوران حقوقی"),
        ("مهندس", 0.14, "مهندسان"),
        ("بازنشسته", 0.05, "بازنشستگان"),
        ("دانشجو", 0.00, "دانشجویان"),
    ]
    cursor.executemany("INSERT INTO job_categories (name, tax_rate, description) VALUES (?, ?, ?)", job_categories)
    
    exemptions = [
        ("معلولیت", 50.0, "افراد دارای معلولیت"),
        ("ایثارگری", 100.0, "ایثارگران و خانواده شهدا"),
        ("سرپرست خانوار", 20.0, "سرپرستان خانوار"),
        ("چند فرزندی", 15.0, "خانواده‌های دارای سه فرزند و بیشتر"),
    ]
    cursor.executemany("INSERT INTO tax_exemptions (name, percentage, description) VALUES (?, ?, ?)", exemptions)
    
    citizens = [
        ("1270765108", "علی", "محمدی", "حسن", "1370-01-15", 1, 1, 850000000),
        ("0023456789", "مریم", "احمدی", "رضا", "1365-05-20", 3, 4, 1500000000),
        ("0034567890", "محمد", "رضایی", "علی", "1380-03-10", 5, 6, 720000000),
        ("0045678901", "زهرا", "حسینی", "محمود", "1355-08-25", 1, 7, 480000000),
        ("0056789012", "امیر", "کریمی", "احمد", "1375-11-30", 4, 3, 950000000),
        ("0067890123", "فاطمه", "موسوی", "جواد", "1360-02-05", 2, 5, 1200000000),
        ("0078901234", "حسین", "علوی", "کاظم", "1385-07-12", 6, 8, 0),
        ("0089012345", "سارا", "نوری", "مهدی", "1368-09-18", 7, 2, 680000000),
    ]
    cursor.executemany('''
        INSERT INTO citizens (national_code, first_name, last_name, father_name, birth_date, city_id, job_category_id, annual_income)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', citizens)
    
    citizen_exemptions = [
        (4, 1, "1400-01-01", "1405-12-29"),
        (1, 3, "1402-01-01", "1406-12-29"),
        (5, 4, "1401-06-01", "1404-06-01"),
    ]
    cursor.executemany('''
        INSERT INTO citizen_exemptions (citizen_id, exemption_id, valid_from, valid_until)
        VALUES (?, ?, ?, ?)
    ''', citizen_exemptions)
    
    conn.commit()
    conn.close()

def find_citizen_by_national_code(national_code):
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            c.id,
            c.national_code,
            c.first_name,
            c.last_name,
            c.father_name,
            c.birth_date,
            c.annual_income,
            c.is_active,
            ct.name as city_name,
            p.name as province_name,
            j.name as job_name,
            j.tax_rate
        FROM citizens c
        LEFT JOIN cities ct ON c.city_id = ct.id
        LEFT JOIN provinces p ON ct.province_id = p.id
        LEFT JOIN job_categories j ON c.job_category_id = j.id
        WHERE c.national_code = ?
    ''', (national_code,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return dict(result)
    return None

def get_citizen_exemptions(citizen_id):
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            e.name,
            e.percentage,
            e.description,
            ce.valid_from,
            ce.valid_until
        FROM citizen_exemptions ce
        JOIN tax_exemptions e ON ce.exemption_id = e.id
        WHERE ce.citizen_id = ?
        AND date(ce.valid_from) <= date('now')
        AND date(ce.valid_until) >= date('now')
    ''', (citizen_id,))
    
    results = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in results]

def save_tax_record(citizen_id, year, income, tax_amount):
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO tax_records (citizen_id, year, income, tax_amount)
        VALUES (?, ?, ?, ?)
    ''', (citizen_id, year, income, tax_amount))
    
    conn.commit()
    conn.close()

def get_tax_history(citizen_id):
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT year, income, tax_amount, is_paid, payment_date, created_at
        FROM tax_records
        WHERE citizen_id = ?
        ORDER BY year DESC
    ''', (citizen_id,))
    
    results = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in results]

init_database()
seed_database()