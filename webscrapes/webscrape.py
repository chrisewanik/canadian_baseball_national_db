# %%
# Script to Scrape CCBC Data

# %%
# Import libraries
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import requests
from bs4 import BeautifulSoup

# %% [markdown]
# Write Functions to Scrape the Table of the Page and Scrape all Tables of the Page Options

# %%
def scrape_batting_table(season_id):
    """Scrape the batting table from the CCBC website for a given season_id

    Args:
        season_id (str): The season_id to scrape the batting table for

    Returns:
        pd.DataFrame: The batting table for the given season_id
    """    
    # Create the path string using the season_id
    path = f'http://pointstreak.com/baseball/stats.html?{season_id}&view=teambatting'
    
    # Get the page
    page = requests.get(path)
    # Create the soup object
    soup = BeautifulSoup(page.content, 'html.parser')
    # Get the table
    table = soup.select('#bat_first > table:nth-child(1)')
    
    # Check if the table is empty
    if not table:
        print(f'No table found for season_id {season_id}')
        return None
    
    # Get the headers (th) from the table
    headers = [th.text.strip() for th in table[0].find_all('th')]
    # Add the season_id to the headers
    headers.append('season_id')
    
    # Create an empty list to store the rows
    rows = []
    
    # Get the rows (tr) from the table
    for tr in table[0].find_all('tr'):
        # Get the data (td) from the row
        row = [td.text.strip() for td in tr.find_all('td')]
        # Check if the row is empty if not append the season_id to the row and append the row to the rows list
        if row:
            row.append(season_id)
            rows.append(row)
    return pd.DataFrame(rows, columns=headers)


# %%
# Function to perform the scraping
def scrape_page(url, scrape_func):
    """Scrape the CCBC website for a given url and scrape_func

    Args:
        url (str): The url to scrape
        scrape_func (function): The function to use to scrape the table

    Returns:
        pd.DataFrame: The dataframe containing the scraped data
    """    
    
    # Get the page
    page = requests.get(url)
    # Create the soup object
    soup = BeautifulSoup(page.content, 'html.parser')
    # Get the select element with the id seasonid
    select = soup.find('select', {'id': 'seasonid'})
    # Get the options from the select element (this is the list of years)
    options = select.find_all('option')
    # Create a dictionary of the years (seasons) with the season_id as the key and the season name as the value
    seasons = {option['value']: option.text for option in options}
    print(seasons)
    
    # For season in season_ids scrape the table and append the results to a dataframe using the scrape_func
    df = pd.DataFrame()
    for season in seasons.keys():
        temp_df = scrape_func(season)
        # append the new row to the DataFrame
        df = pd.concat([df, temp_df])

    # Create a new variable called season that uses the season_id column to look up the season name in the seasons dictionary
    df['season'] = df['season_id'].map(seasons)

    # Drop the season_id column
    df.drop('season_id', axis=1, inplace=True)
    
    # Return the dataframe
    return df
    

# %% [markdown]
# Scrape Team Batting Stats

# %%
# Declare the URL
url = 'http://pointstreak.com/baseball/stats.html?leagueid=160&seasonid=186&view=teambatting'

path = 'http://pointstreak.com/baseball/stats.html?leagueid=160&seasonid={season_id}&view=teambatting'

# Call the scrape_page function and pass in the url
batting_df = scrape_page(url, scrape_batting_table)

# Preview the dataframe
batting_df.head()

# Save as a csv file
batting_df.to_csv('ccbc_batting.csv', index=False)

# %%
def scrape_pitching_table(season_id):
    """Scrape the pitching table from the CCBC website for a given season_id

    Args:
        season_id (str): The season_id to scrape the pitching table for (this is the year)

    Returns:
        pd.DataFrame: The pitching table for the given season_id
    """    
    
    # Create the path string using the season_id
    path = f'http://pointstreak.com/baseball/stats.html?{season_id}&view=teampitching'

    # Get the page
    page = requests.get(path)
    # Create the soup object
    soup = BeautifulSoup(page.content, 'html.parser')
    # Get the table
    table = soup.select('#pitch_first > table:nth-child(1)')
    if not table:
        print(f'No table found for season_id {season_id}')
        return None

    # Get the headers from the table (th)
    headers = [th.text.strip() for th in table[0].find_all('th')]
    # Add the season_id to the headers
    headers.append('season_id')
    
    # Create an empty list to store the rows
    rows = []

    # Get the rows from the table (tr)    
    for tr in table[0].find_all('tr'):
        # Get the data from the row (td)
        row = [td.text.strip() for td in tr.find_all('td')]
        # Check if the row is empty if not append the season_id to the row and append the row to the rows list
        if row:
            row.append(season_id)
            rows.append(row)
    return pd.DataFrame(rows, columns=headers)


# %% [markdown]
# Scrape Team Pitching Stats

# %%
# Declare the URL
url = 'http://pointstreak.com/baseball/stats.html?leagueid=160&seasonid=186&view=teampitching'

# Call the scrape_page function and pass in the url
pitching_df = scrape_page(url, scrape_pitching_table)

# Preview the dataframe
pitching_df.head()

# Save as a csv file
pitching_df.to_csv('ccbc_pitching.csv', index=False)

# %% [markdown]
# Scrape the Standings

# %%
def scrape_standings_table(season_id):
    """Scrape the standings table from the CCBC website for a given season_id

    Args:
        season_id (str): The season_id to scrape the standings table for (this is the year)

    Returns:
        pd.DataFrame: The standings table for the given season_id
    """    
    # Create the path string using the season_id
    path = f'http://pointstreak.com/baseball/standings.html?{season_id}&stype=l'
    # Get the page
    page = requests.get(path)
    # Create the soup object
    soup = BeautifulSoup(page.content, 'html.parser')
    # Get the table
    table = soup.select('#psbb_standings > table:nth-child(1)')
    if not table:
        print(f'No table found for season_id {season_id}')
        return None

    # Get the headers from the table (th)
    headers = [th.text.strip() for th in table[0].find_all('th')]
    # Add the season_id to the headers
    headers.append('season_id')
    
    # Create an empty list to store the rows
    rows = []

    # Get the rows from the table (tr)    
    for tr in table[0].find_all('tr'):
        # Get the data from the row (td)
        row = [td.text.strip() for td in tr.find_all('td')]
        if row:
            row.append(season_id)
            rows.append(row)
    return pd.DataFrame(rows, columns=headers)


# %%
# Declare the URL
url = 'http://pointstreak.com/baseball/standings.html?leagueid=160&seasonid=186&stype=l'

# Call the scrape_page function and pass in the url
standings_df = scrape_page(url, scrape_standings_table)

# Preview the dataframe
standings_df.head()

# Save as a csv file
standings_df.to_csv('ccbc_standings.csv', index=False)

# %% [markdown]
# Scrape Qualified Batters

# %%
def scrape_qual_batters_table(season_id):
    """Scrape the qualified batters table from the CCBC website for a given season_id

    Args:
        season_id (str): The season_id to scrape the qualified batters table for (this is the year)

    Returns:
        pd.DataFrame: The qualified batters table for the given season_id
    """    
    
    # Create the path string using the season_id
    path = f'http://pointstreak.com/baseball/stats.html?{season_id}&view=batting'
    # Get the page
    page = requests.get(path)
    # Create the soup object
    soup = BeautifulSoup(page.content, 'html.parser')
    # Get the table
    table = soup.select('#battingresults')
    if not table:
        print(f'No table found for season_id {season_id}')
        return None

    # Get the headers from the table (th)
    headers = [th.text.strip() for th in table[0].find_all('th')]
    # Add the season_id to the headers
    headers.append('season_id')
    
    # Create an empty list to store the rows
    rows = []

    # Get the rows from the table (tr)    
    for tr in table[0].find_all('tr'):
        # Get the data from the row (td)
        row = [td.text.strip() for td in tr.find_all('td')]
        if row:
            row.append(season_id)
            rows.append(row)
    return pd.DataFrame(rows, columns=headers)


# %%
# Declare the URL
url = 'http://pointstreak.com/baseball/stats.html?leagueid=160&seasonid=186&view=batting'

# Call the scrape_page function and pass in the url
df = scrape_page(url, scrape_qual_batters_table)

# Save as a csv file
df.to_csv('ccbc_qual_batters.csv', index=False)

# %% [markdown]
# Scrape Qualified Pitchers

# %%
def scrape_qual_pitchers_table(season_id):
    """Scrape the qualified pitchers table from the CCBC website for a given season_id

    Args:
        season_id (str): The season_id to scrape the qualified pitchers table for (this is the year)

    Returns:
        pd.DataFrame: The qualified pitchers table for the given season_id
    """    
    
    # Create the path string using the season_id
    path = f'http://pointstreak.com/baseball/stats.html?{season_id}&view=pitching'
    # Get the page
    page = requests.get(path)
    # Create the soup object
    soup = BeautifulSoup(page.content, 'html.parser')
    # Get the table
    table = soup.select('#pitchingresults')
    if not table:
        print(f'No table found for season_id {season_id}')
        return None
    # Get the headers from the table (th)
    headers = [th.text.strip() for th in table[0].find_all('th')]
    # Add the season_id to the headers
    headers.append('season_id')
    
    # Create an empty list to store the rows
    rows = []

    # Get the rows from the table (tr)    
    for tr in table[0].find_all('tr'):
        # Get the data from the row (td)
        row = [td.text.strip() for td in tr.find_all('td')]
        if row:
            row.append(season_id)
            rows.append(row)
    return pd.DataFrame(rows, columns=headers)


# %%
# Declare the URL
url = 'http://pointstreak.com/baseball/stats.html?leagueid=160&seasonid=186&view=pitching'

# Call the scrape_page function and pass in the url
df = scrape_page(url, scrape_qual_pitchers_table)

# Save as a csv file
df.to_csv('ccbc_qual_pitchers.csv', index=False)


