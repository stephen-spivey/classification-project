host = 'data.codeup.com'
username = 'tobias_2308'
password = 'ih1LPzX44btoIImWWG5Y1WWvPjYfKhYA'

def get_db_url(database_name):
    """
    
    """
    url = f'mysql+pymysql://{username}:{password}@{host}/{database_name}'

    return url