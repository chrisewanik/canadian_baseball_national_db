# %%
# ccbc_db_create


# %%
# Import psycopg2
import psycopg2

# %%
# Connect to the database
try:
    conn = psycopg2.connect("host = localhost dbname = ccbc_db user = postgres password = 123456")
    conn.autocommit = True # This automatically commits any changes to the database without having to call conn.commit() after each command.
    cur = conn.cursor()
except Exception as e:
    print("Unable to connect to the database")
    raise Exception(e)
else:
    print("Database connected")

# %%
# Drop Tables before creating them
try:
    cur.execute("""DROP TABLE IF EXISTS player CASCADE;
                DROP TABLE IF EXISTS player_batting CASCADE;
                DROP TABLE IF EXISTS player_pitching CASCADE;
                DROP TABLE IF EXISTS team CASCADE;
                DROP TABLE IF EXISTS team_batting CASCADE;
                DROP TABLE IF EXISTS team_pitching CASCADE;
                DROP TABLE IF EXISTS standings CASCADE;
                """)
except Exception as e:
    print(e)
    exit()
conn.commit()

# %%
# Create a table called team
try:
    cur.execute("""
        CREATE TABLE team (
            tid SERIAL PRIMARY KEY,
            team_abbr VARCHAR(10),
            year NUMERIC,
            season_type VARCHAR(20),
            UNIQUE (team_abbr, year, season_type)
            );
        """)
    conn.commit()
except Exception as e:
    print(e)
    exit()
else:
    print("Table created")

# %%
# Create a table called player
try:
    cur.execute("""
        CREATE TABLE player (
            pid SERIAL PRIMARY KEY,
            last_name VARCHAR(100),
            first_initial varchar(10),
            team_abbr VARCHAR(10),
            year NUMERIC            
        );
        """)
    conn.commit()
except Exception as e:
    print(e)
    exit()
else:
    print("Table created")

# %%
# Create a table called standings
try:
    cur.execute("""
        CREATE TABLE standings (
            team_abbr VARCHAR(10),
            year NUMERIC,
            season_type VARCHAR(20),
            GP NUMERIC,
            W NUMERIC,
            L NUMERIC,
            PTS NUMERIC,
            PCT NUMERIC,
            PRIMARY KEY (team_abbr, year, season_type)
            );
        """)
    conn.commit()
except Exception as e:
    print(e)
    exit()
else:
    print("Table created")

# %%
# Create a table called team_batting
try:
    cur.execute("""
        CREATE TABLE team_batting (
            team_abbr VARCHAR(10),
            year NUMERIC,
            season_type VARCHAR(20),
            G NUMERIC,
            AB NUMERIC,
            R NUMERIC,
            H NUMERIC,
            Doubles NUMERIC,
            Triples NUMERIC,
            HR NUMERIC,
            RBI NUMERIC,
            TB NUMERIC,
            BB NUMERIC,
            PRIMARY KEY (team_abbr, year, season_type)
            );
        """)
    conn.commit()
except Exception as e:
    print(e)
    exit()
else:
    print("Table created")

# %%
# Create a table called team_pitching
['team_abbr', 'W', 'L', 'IP', 'R', 'ER', 'H', 'BB', 'WP', 'HBP', 'SO', 'BF', 'year', 'season_type']
try:
    cur.execute("""
        CREATE TABLE team_pitching (
            team_abbr VARCHAR(10),
            year NUMERIC,
            season_type VARCHAR(20),
            W NUMERIC,
            L NUMERIC,
            IP NUMERIC,
            R NUMERIC,
            ER NUMERIC,
            H NUMERIC,
            BB NUMERIC,
            WP NUMERIC,
            HBP NUMERIC,
            SO NUMERIC,
            BF NUMERIC,
            PRIMARY KEY (team_abbr, year, season_type)
            );
        """)
    conn.commit()
except Exception as e:
    print(e)
    exit()
else:
    print("Table created")

# %%
# Create a table called player_batting

try:
    cur.execute("""
    CREATE TABLE player_batting (
        last_name VARCHAR(100),
        first_initial varchar(10),
        team_abbr VARCHAR(10),
        year NUMERIC,
        season_type VARCHAR(20),
        P varchar(10),
        AVG NUMERIC,
        G NUMERIC,
        AB NUMERIC,
        R NUMERIC,
        H NUMERIC,
        Doubles NUMERIC,
        Triples NUMERIC,
        HR NUMERIC,
        RBI NUMERIC,
        BB NUMERIC,
        HBP NUMERIC,
        SO NUMERIC,
        SF NUMERIC,
        SH NUMERIC,
        SB NUMERIC,
        CS NUMERIC,
        DP NUMERIC,
        E NUMERIC,
        PRIMARY KEY (last_name, first_initial, year, season_type)
    );
        """)
    conn.commit()
except Exception as e:
    print(e)
    exit()
else:
    print("Table created")

# %%
# Create a table called player_pitching

try:
    cur.execute("""
    CREATE TABLE player_pitching (
        last_name VARCHAR(100),
        first_initial varchar(10),
        team_abbr VARCHAR(10),
        year NUMERIC,
        season_type VARCHAR(20),
        G NUMERIC,
        GS NUMERIC,
        CG NUMERIC,
        IP NUMERIC,
        H NUMERIC,
        R NUMERIC,
        ER NUMERIC,
        BB NUMERIC,
        SO NUMERIC,
        W NUMERIC,
        L NUMERIC,
        SV NUMERIC,
        Doubles NUMERIC,
        Triples NUMERIC,
        ERA NUMERIC,
        PRIMARY KEY (last_name, first_initial, year, season_type)
    );
        """)
    conn.commit()
except Exception as e:
    print(e)
    exit()
else:
    print("Table created")


