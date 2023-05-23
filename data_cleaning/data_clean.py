# %%
# Import Libraries
import numpy as np
import pandas as pd
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

# %% [markdown]
# Goal 1: Create the Players Table which should be the pid, tid, lname and finitial of the player 

# %%
# Read in the ccbc_qual_batters and ccbc_qual_pitchers data
ccbc_qual_batters_df = pd.read_csv('ccbc_qual_batters.csv')
ccbc_qual_pitchers_df = pd.read_csv('ccbc_qual_pitchers.csv')

# %%
# Vertically Stack the DataFrames
ccbc_df = pd.concat([ccbc_qual_batters_df, ccbc_qual_pitchers_df], axis=0)

# %%
# Keep the Player Name, Team, and season columns
player_prelim_df = ccbc_df[['Player', 'Team', 'season']]

# %%
# Split the Player column into last_name and first_initial on the comma
player_prelim_df[['last_name', 'first_initial']] = player_prelim_df['Player'].str.split(',', expand=True)

# %%
# Extract the numeric portion of the season column
player_prelim_df['year'] = player_prelim_df['season'].str.extract('(\d+)')

# %%
# Extract the string portion of the season column
player_prelim_df['season_type'] = player_prelim_df['season'].str.extract('([A-Za-z]+)')

# %%
# Get the unique values of the season_type column
player_prelim_df['season'].unique()

# %%
# Create a variable, championships, that is true when season_type contains the string 'Champ'
player_prelim_df['championships'] = player_prelim_df['season'].str.contains('Champ')

# %%
# If season_type is CCBC and championships is true, then set season_type to CCBC Champ
player_prelim_df.loc[(player_prelim_df['season_type'] == 'CCBC') & (player_prelim_df['championships'] == True), 'season_type'] = 'CCBC Champ'

# %%
#  Create a new df players that is that is a unique list of players and their team and year
players = player_prelim_df[['last_name', 'first_initial', 'Team', 'year', 'season_type']].drop_duplicates()

# %%
# create pid column that is a unique identifier for each player and make it the first column
players.reset_index(inplace=True)
players['pid'] = players.index + 1

# %%
# Change the order of the columns so that pid is the first column
players = players[['pid', 'last_name', 'first_initial', 'Team', 'year']]

# %%
# rename Team to team_abbr
players.rename(columns={'Team': 'team_abbr'}, inplace=True)

# %%
# Find where the player index is 42 and change it to 0
players.loc[players['pid'] == 42, 'pid'] = 0 # In Honor of Jackie Robinson <3

# %%
players.head()

# %%
# Get the length of the players df
len(players)

# %%
# Get the number of unique pid values
len(players['pid'].unique())

# %% [markdown]
# Goal 2: Create the Teams Table which should be the tid, tname, year and season

# %%
# Create a new df called teams that is the team, year, season_type, and tid (which is a unique identifier for each team, year) pulled from the players_pre_df
teams = player_prelim_df[['Team', 'year', 'season_type']].drop_duplicates()
teams['tid'] = teams.index + 1

# Rename Team to team_abbr
teams.rename(columns={'Team': 'team_abbr'}, inplace=True)

# Remove any duplicate rows based on the tid column
teams = teams.drop_duplicates(subset=['tid'])

teams.head()

# %%
teams.columns

# %%
# Get the number of unique values of the tid column
teams['tid'].nunique()

# %%
# Get the number of rows in the teams df
teams.shape[0]

# %% [markdown]
# 3. Create the Standings Table which should be the team_abbr, year, season, wins, losses, and winpct

# %%
# Load in ccbc_standings.csv
ccbc_standings_df = pd.read_csv('ccbc_standings.csv')

# %%
ccbc_standings_df.head()

# %%
# Get all unique teams in the teams table (to mutate the teams_abbr column)
teams['team_abbr'].unique()

# %%
# Get the number of unique teams
len(ccbc_standings_df['Team'].unique()) # 117

# Get all unique teams
ccbc_standings_df['Team'].unique()

# %%
# Create the teams_dict which maps the team name to the team abbreviation

teams_dict = {
    'Prairie Baseball Academy': 'PBA',
    'Dinos': 'UC',
    'Blue Mountain CC': 'BMC',
    'College of Southern Nevada': 'CSN',
    'Colorado Northern CC': 'CNC',
    'University of Calgary': 'UC',
    'Okanagan College': 'OC',
    'Victoria Collegiate': 'VC',
    'Thompson Rivers University': 'TRU',
    'University of Fraser Valley': 'UFV',
    'Edmonton Collegiate': 'EC',
    'Vancouver Island University': 'VIU',
    'Salt Lake CC': 'SLC',
    'Walla Walla CC': 'WWC',
    'Douglas College': 'DC',
    'Eastern Utah State': 'EUS',
    'Glendale CC': 'GCC',
    'Northeaster JC': 'NEJ',
    'Phoenix College': 'PC',
    'Pima CC': 'PCC',
    'Spokane Falls CC': 'SFC',
    'UFV': 'UFV',
    'University of Alberta': 'UOA',
    'Utah State University Eastern': 'USU',
    'Vancouver Island': 'VIU',
    'Yavapai CC': 'YCC',
    'University of the Fraser Valley': 'UFV',
    'Columbia Basin CC': 'CBC',
    'College of Southern Idaho': 'CSI',
    'Chandler Gilbert': 'CGC',
    'CCBC Championships': 'CBT',
    'Bristol University': 'BU',
    'Arizona Christian': 'AC',
    'Providence Christian College': 'PAS',
    "St. Katherine's": 'STK',
    'Van. Island Baseball Institute': 'VIU',
    'Eastern Arizona College': 'EAC',
    'Everett CC': 'EVC',
    'CNCC': 'CNC',
    'Shorline Community College': 'SCC',
    'St. Katherine': 'STK',
    'Northwest Nazarene': 'NN',
    'Cochise CC': 'CCC',
    'Chandler/Gilbert CC': 'CGC',
    'UofA Club': 'UOA',
    'Mesa CC': 'MC',
    'Paradise Valley CC': 'PAS',
    'So. Mountain CC': 'SMC',
    'Langley Blaze': 'LB',
    'Olympic CC': 'OLY',
    'Abbotsford': 'AB',
    'Big Bend CC': 'BB',
    'Green River CC': 'GRE',
    'USUE CC': 'CEU',
    'Clark CC': 'CLK',
    'Grand Canyon University': 'GCU',
    'College of S.Idaho': 'CSI',
    'Spokane Falls': 'SFC',
    'Treasure Valley CC': 'TVC',
    'Chemeketa CC': 'CCC',
    'Western Nevada': 'WN',
    'CWI': 'CWI',
    'Edmonds CC': 'ECC',
    'Mt. Hood CC': 'MHC',
    'Southwest Oregon': 'SWO',
    'Western Washington': 'DSU',
    'Columbia Basin': 'CBC',
    'Wenatchee CC': 'WCC',
    'Linn-Benton CC': 'LB',
    'UC': 'UC',
    'Dawgs': 'PBA',
    'WolfPack': 'TRU',
    'Coyotes': 'OC',
    'NP': 'NP',
    'Eastern Utah CC': 'EUS',
    'Dixie State University': 'DSU',
    'Clackamas CC': 'CLK',
    'College of Eastern Utah': 'CEU',
    'TVCC Tournament': 'TVC',
    'UYA Academy All Stars': 'UYA',
    'Arizona Western CC': 'AWC',
    'Nanaimo Pirates': 'NP',
    'Victoria Mariners': 'VICMs',
    'Mariners': 'VIU',
    'Victoria Eagles': 'VC',
    'Shoreline CC': 'SHO',
    'Skagit Valley CC': 'SVC',
    'Olympic College': 'OLY',
    'Arizona Christian College': 'ACC',
    'Central Arizona College': 'CAC',
    'Chandler Gilbert CC': 'CGC',
    'Gateway CC': 'GCC',
    'Columbia Basin Tournament': 'CBT',
    'Pierce College': 'PC',
    'Salt Lake': 'SLC',
    'CSI Idaho': 'CSI',
    'UCC-TRU Alumni': 'Wal',
    'San Diego CC': 'SCC',
    'Pasco': 'PAS',
    'UYA All-Stars': 'UYA',
    'Miles City CC': 'Mil',
    'Victoria Mavericks Baseball': 'BB',
    'Grays Harbor': 'GRE',
    'Douglas CC': 'DC',
    'Pierce CC': 'PCC',
    'Chandler-Gilbert CC': 'CGC',
    'Miles CC': 'Mil',
    'TRU WolfPack': 'TRU',
    'OC Baseball': 'OC',
    'Bellevue CC': 'Bel',
    'CS Idaho': 'CSI',
    'Blue Mtn. CC': 'BM',
    'PBA Dawgs': 'PBA',
    'U of Arizona Club': 'UOA',
    'Texas Tech Club': 'TT',
    'Ontario Blue Jays': 'OBJ',
    'OCC': 'OC',
    'RBI All-Stars': 'SFC',
    'Urban Youth All-Stars': 'UYA',
    'Urban Youth  All-Stars': 'UYA',
    'Arizona State Club Team': 'ASU',
    'UofC Dinos': 'UC',
    'Grays Harbour CC': 'GRE',
    'VIBI Mariners': 'VIU',
    'KPU Eagles': 'KPU',
    'Kwantlen Polytechnic University': 'KPC',
    'UofC': 'UC',
    'UoC': 'UC',
    'Coyotes': 'OC',
    'Mariners': 'VIU',
    'Dawgs': 'PBA',
    ' Dawgs': 'PBA',
    ' Mariners': 'VIU'
}

# %%
# Use the teams_dict to create a new column in the ccbc_standings_df called 'team_abbr'
ccbc_standings_df['team_abbr'] = ccbc_standings_df['Team'].map(teams_dict)

# %%
# See the unique list of teams who have no abbreviation
ccbc_standings_df[ccbc_standings_df['team_abbr'].isnull()]['Team'].unique() # Success!

# %%
ccbc_standings_df.head()

# %%
# Create a new df called standings_df that only contains GP, W, L, PTS, PCT team_abbr, season
standings_df = ccbc_standings_df[['GP', 'W', 'L', 'PTS', 'PCT', 'team_abbr', 'season']]

# %%
# Create the season_type column
# Extract the numeric portion of the season column
standings_df['year'] = standings_df['season'].str.extract('(\d+)')

# Extract the string portion of the season column
standings_df['season_type'] = standings_df['season'].str.extract('([A-Za-z]+)')

# Create a variable, championships, that is true when season_type contains the string 'Champ'
standings_df['championships'] = standings_df['season'].str.contains('Champ')

# I season_type is CCBC and championships is true, then set season_type to CCBC Champ
standings_df.loc[(standings_df['season_type'] == 'CCBC') & (standings_df['championships'] == True), 'season_type'] = 'CCBC Champ'

standings_df.head()

# %%
# Drop the column season and championships
standings_df.drop(columns=['championships'], inplace=True)

# %%
# Join the standings_df to the teams dataframe where the team_abbr matches the Team column
standings_df = standings_df.merge(teams, how='left', left_on=['team_abbr'], right_on=['team_abbr'])

# %%
# Rename year_x to year, season_type_x to season_type, and drop the year_y and season_type_y columns
standings_df.rename(columns={'year_x': 'year', 'season_type_x': 'season_type'}, inplace=True)
standings_df.drop(columns=['year_y', 'season_type_y'], inplace=True)

# %%
standings_df.head()

# %%
# Reorder the standings_df columns: team_abbr, season, year, season_type, GP, W, L, PTS, PCT
standings_df = standings_df[['team_abbr', 'season', 'year', 'season_type', 'GP', 'W', 'L', 'PTS', 'PCT']]

# %%
# Get rows with null values
standings_df[standings_df.isnull().any(axis=1)]

# %%
# 
# standings_df.reset_index(drop=True, inplace=True)

# %%
# Drop duplicates from the standings_df
standings_df.drop_duplicates(subset=['team_abbr', 'season', 'year', 'season_type'], keep='first', inplace=True)

# %%
# Reset the index
standings_df.reset_index(inplace=True)

# %%
# Drop the season, level_0 and index column
standings_df.drop(columns=['season', 'index'], inplace=True)

# %%
# Print the standings_df where the season type is CCBC and the year is 2017
standings_df[(standings_df['season_type'] == 'CCBC') & (standings_df['year'] == '2017')]

# %%
# Print the standings_df where the season type is CCBC and the year is 2017
standings_df[(standings_df['season_type'] == 'CCBC Champ') & (standings_df['year'] == '2017')]

# %%
standings_df.columns

# %%
# Get unique values for the team_abbr column
standings_df['team_abbr'].unique()

# %% [markdown]
# 4. Create the team_pitching dataframe

# %%
# Load in ccbc_pitching.csv
ccbc_pitching_df = pd.read_csv('ccbc_pitching.csv')

# %%
# Rename Team to team_abbr
ccbc_pitching_df.rename(columns={'Team': 'team_abbr'}, inplace=True)

# %%
# Create team_pitching_df that is a copy of ccbc_pitching_df
team_pitching_df = ccbc_pitching_df.copy()

# Create the season_type column
# Extract the numeric portion of the season column
team_pitching_df['year'] = team_pitching_df['season'].str.extract('(\d+)')

# Extract the string portion of the season column
team_pitching_df['season_type'] = team_pitching_df['season'].str.extract('([A-Za-z]+)')

# Create a variable, championships, that is true when season_type contains the string 'Champ'
team_pitching_df['championships'] = team_pitching_df['season'].str.contains('Champ')

# I season_type is CCBC and championships is true, then set season_type to CCBC Champ
team_pitching_df.loc[(team_pitching_df['season_type'] == 'CCBC') & (team_pitching_df['championships'] == True), 'season_type'] = 'CCBC Champ'

# drop the season and championships columns
team_pitching_df.drop(columns=['season', 'championships'], inplace=True)

# Use the teams_dict to create a new column in the team_batting_df called 'team_abbr' and if the team_abbr is not in the teams_dict, then set it to do nothing
team_pitching_df['team_abbr'] = team_pitching_df['team_abbr'].map(teams_dict).fillna(team_pitching_df['team_abbr'])

team_pitching_df.head()

# %%
# Remove the rows where the index is 110 and 226 from team_pitching_df
team_pitching_df.drop(index=[110, 226], inplace=True)

# %%
team_pitching_df.shape

# %%
# Remove the rows where team_abbr is Total:
team_pitching_df = team_pitching_df[team_pitching_df['team_abbr'] != 'Totals:']

# %%
team_pitching_df.shape

# %%
team_pitching_df.columns

# %% [markdown]
# 5. Create the team_batting dataframe

# %%
# Load in ccbc_batting.csv
ccbc_batting_df = pd.read_csv('ccbc_batting.csv')

# %%
# Rename Team to team_abbr
ccbc_batting_df.rename(columns={'Team': 'team_abbr'}, inplace=True)

# %%
# Create team_batting_df that is a copy of ccbc_batting_df
team_batting_df = ccbc_batting_df.copy()

# Create the season_type column
# Extract the numeric portion of the season column
team_batting_df['year'] = team_batting_df['season'].str.extract('(\d+)')

# Extract the string portion of the season column
team_batting_df['season_type'] = team_batting_df['season'].str.extract('([A-Za-z]+)')

# Create a variable, championships, that is true when season_type contains the string 'Champ'
team_batting_df['championships'] = team_batting_df['season'].str.contains('Champ')

# I season_type is CCBC and championships is true, then set season_type to CCBC Champ
team_batting_df.loc[(team_batting_df['season_type'] == 'CCBC') & (team_batting_df['championships'] == True), 'season_type'] = 'CCBC Champ'

# drop the season and championships columns
team_batting_df.drop(columns=['season', 'championships'], inplace=True)

# Use the teams_dict to create a new column in the team_batting_df called 'team_abbr' and if the team_abbr is not in the teams_dict, then set it to do nothing
team_batting_df['team_abbr'] = team_batting_df['team_abbr'].map(teams_dict).fillna(team_batting_df['team_abbr'])

# Reorder so that year and season_type are after team_abbr
team_batting_df = team_batting_df[['team_abbr', 'year', 'season_type','G','AB','R','H','2B','3B','HR','RBI','TB','BB']]

team_batting_df.head()

# %%
# Get the 2017 CCBC batting stats
team_batting_df[(team_batting_df['year'] == '2012') & (team_batting_df['season_type'] == 'Pre')]

# %%
# Drop Index 116, 117, 227
team_batting_df.drop(index=[116, 117, 227], inplace=True)
team_batting_df.reset_index(inplace=True)

# %%
# Remove the rows where team_abbr is Total:
team_batting_df = team_batting_df[team_batting_df['team_abbr'] != 'Total:']

# %% [markdown]
# 6. Create the player_batting dataframe

# %%
# Load in the ccbc_qual_batters.csv
ccbc_qual_batters_df = pd.read_csv('ccbc_qual_batters.csv')

# %%
# Create team_batting_df that is a copy of ccbc_batting_df
player_batting_df = ccbc_qual_batters_df.copy()

# Split the Player column on the comma into last_name and first_initial
player_batting_df[['last_name', 'first_initial']] = player_batting_df['Player'].str.split(',', expand=True)

# Rename Team to team_abbr
player_batting_df.rename(columns={'Team': 'team_abbr'}, inplace=True)

# Create the season_type column
# Extract the numeric portion of the season column
player_batting_df['year'] = player_batting_df['season'].str.extract('(\d+)')

# Extract the string portion of the season column
player_batting_df['season_type'] = player_batting_df['season'].str.extract('([A-Za-z]+)')

# Create a variable, championships, that is true when season_type contains the string 'Champ'
player_batting_df['championships'] = player_batting_df['season'].str.contains('Champ')

# I season_type is CCBC and championships is true, then set season_type to CCBC Champ
player_batting_df.loc[(player_batting_df['season_type'] == 'CCBC') & (player_batting_df['championships'] == True), 'season_type'] = 'CCBC Champ'

# drop the season and championships columns
player_batting_df.drop(columns=['season', 'championships', 'No Data', 'Player'], inplace=True)

# Reorder so that year and season_type are after team_abbr
player_batting_df = player_batting_df[['first_initial', 'last_name', 'team_abbr', 'year', 'season_type', 'P', 'AVG', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB',
       'HBP', 'SO', 'SF', 'SH', 'SB', 'CS', 'DP', 'E']]

player_batting_df.head()

# %%
player_batting_df.columns

# %% [markdown]
# 7. Create the player_pitching dataframe

# %%
# Load in the ccbc_qual_pitchers.csv
ccbc_qual_pitchers_df = pd.read_csv('ccbc_qual_pitchers.csv')

# %%
# Create team_batting_df that is a copy of ccbc_batting_df
player_pitching_df = ccbc_qual_pitchers_df.copy()

# Split the Player column on the comma into last_name and first_initial
player_pitching_df[['last_name', 'first_initial']] = player_pitching_df['Player'].str.split(',', expand=True)

# Rename Team to team_abbr
player_pitching_df.rename(columns={'Team': 'team_abbr'}, inplace=True)

# Create the season_type column
# Extract the numeric portion of the season column
player_pitching_df['year'] = player_pitching_df['season'].str.extract('(\d+)')

# Extract the string portion of the season column
player_pitching_df['season_type'] = player_pitching_df['season'].str.extract('([A-Za-z]+)')

# Create a variable, championships, that is true when season_type contains the string 'Champ'
player_pitching_df['championships'] = player_pitching_df['season'].str.contains('Champ')

# I season_type is CCBC and championships is true, then set season_type to CCBC Champ
player_pitching_df.loc[(player_pitching_df['season_type'] == 'CCBC') & (player_pitching_df['championships'] == True), 'season_type'] = 'CCBC Champ'

# drop the season and championships columns
player_pitching_df.drop(columns=['season', 'championships', 'No Data', 'Player'], inplace=True)

# Reorder so that year and season_type are after team_abbr
player_pitching_df = player_pitching_df[['first_initial', 'last_name', 'team_abbr', 'year',
       'season_type', 'G', 'GS', 'CG', 'IP', 'H', 'R', 'ER', 'BB', 'SO', 'W', 'L',
       'SV', '2B', '3B', 'ERA']]

player_pitching_df.head()

# %%
# Get the player with the most ever strikeouts in a single season
player_pitching_df.loc[player_pitching_df['SO'].idxmax()]

# %%
# Get the columns of the players df
player_pitching_df.columns

# %% [markdown]
# Insert Data into the Database

# %%
# Insert the teams data into the team table
for index, row in teams.iterrows():
    try:
        cur.execute("INSERT INTO team (tid, team_abbr, year, season_type) VALUES (%s, %s, %s, %s)", (row['tid'], row['team_abbr'], row['year'], row['season_type']))
    except Exception as e:
        print("Unable to insert row")
        raise Exception(e)
    # else:
    #     print("Row inserted")

# %%
# Insert the players data into the player table
for index, row in players.iterrows():
    try:
        cur.execute("INSERT INTO player (pid, last_name, first_initial, team_abbr, year) VALUES (%s, %s, %s, %s, %s)", (row['pid'], row['last_name'], row['first_initial'], row['team_abbr'], row['year']))
    except Exception as e:
        print("Unable to insert row")
        raise Exception(e)
    # else:
    #     print("Row inserted")

# %%
# Insert the data from the standings_df into the standings table
for index, row in standings_df.iterrows():
    try:
        cur.execute("INSERT INTO standings (team_abbr, year, season_type, GP, W, L, PTS, PCT) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", 
                    (row['team_abbr'], row['year'], row['season_type'], row['GP'], row['W'], row['L'], row['PTS'], row['PCT']))
    except Exception as e:
        print("Unable to insert row")
        raise Exception(e)
    # else:
    #     print("Row inserted")
    

# %%
# Insert the data from the team_pitching_df into the team_pitching table
for index, row in team_pitching_df.iterrows():
    try:
        cur.execute("INSERT INTO team_pitching (team_abbr, year, season_type, W, L, IP, R, ER, H, BB, WP, HBP, SO, BF) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s,%s,%s)", 
                    (row['team_abbr'], row['year'], row['season_type'], row['W'], row['L'], row['IP'], row['R'], row['ER'], row['H'], row['BB'], row['WP'], row['HBP'], row['SO'], row['BF']))
    except psycopg2.errors.UniqueViolation as u:
            # Print the row that caused the error
            print(row)
            # If a primary or foreign key constraint is violated, skip row and continue
            conn.rollback()
            continue
    except Exception as e:
        print("Unable to insert row")
        raise Exception(e)
    # else:
    #     print("Row inserted")

# %%
team_batting_df.columns

# %%
# Insert the data from the team_batting_df into the team_batting table
error_dict = {}
for index, row in team_batting_df.iterrows():
    try:
        cur.execute("INSERT INTO team_batting (team_abbr, year, season_type, G, AB, R, H, Doubles, Triples, HR, RBI, TB, BB) VALUES (%s,%s,%s,%s,%s,%s,%s,%s, %s, %s, %s, %s, %s)",
                    (row['team_abbr'], row['year'], row['season_type'], row['G'], row['AB'], row['R'], row['H'], row['2B'], row['3B'], row['HR'], row['RBI'], row['TB'], row['BB']))
    except psycopg2.errors.UniqueViolation as u:
            # Print the row that caused the error
            # print(row)
            # Append the row to the error dictionary
            error_dict[index] = row
            # If a primary or foreign key constraint is violated, skip row and continue
            conn.rollback()
            continue
    except Exception as e:
        print("Unable to insert row")
        raise Exception(e)
    # else:
    #     print("Row inserted")  

# %%
# Insert the data from the player_batting_df into the player_batting table
error_dict = {}
for index, row in player_batting_df.iterrows():
    try:
        cur.execute("INSERT INTO player_batting (first_initial, last_name, team_abbr, year, season_type, P, AVG, G, AB, R, H, Doubles, Triples, HR, RBI, BB, HBP, SO, SF, SH, SB, CS, DP, E) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (row['first_initial'], row['last_name'], row['team_abbr'], row['year'], row['season_type'], row['P'], row['AVG'], row['G'], row['AB'], row['R'],
                     row['H'], row['2B'], row['3B'], row['HR'], row['RBI'], row['BB'], row['HBP'], row['SO'], row['SF'], row['SH'], row['SB'], row['CS'], row['DP'], row['E']))
    except psycopg2.errors.UniqueViolation as u:
            # Print the row that caused the error
            # print(row)
            # Append the row to the error dictionary
            error_dict[index] = row
            # If a primary or foreign key constraint is violated, skip row and continue
            conn.rollback()
            continue
    except Exception as e:
        print("Unable to insert row")
        raise Exception(e)
    # else:
    #     print("Row inserted")   

# %%
# Insert the data from the player_pitching_df into the player_pitching table
error_dict = {}
for index, row in player_pitching_df.iterrows():
    try:
        cur.execute("INSERT INTO player_pitching (first_initial, last_name, team_abbr, year, season_type, G, GS, CG, IP, H, R, ER, BB, SO, W, L, SV, Doubles, Triples, ERA) VALUES (%s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s,%s)",
                    (row['first_initial'], row['last_name'], row['team_abbr'], row['year'], row['season_type'], row['G'], row['GS'], row['CG'], row['IP'], row['H'], row['R'], row['ER'], row['BB'], row['SO'], row['W'], row['L'], row['SV'], row['2B'], row['3B'], row['ERA']))
    except psycopg2.errors.UniqueViolation as u:
            # Print the row that caused the error
            # print(row)
            # Append the row to the error dictionary
            error_dict[index] = row
            # If a primary or foreign key constraint is violated, skip row and continue
            conn.rollback()
            continue
    except Exception as e:
        print("Unable to insert row")
        raise Exception(e)
    # else:
    #     print("Row inserted")   


