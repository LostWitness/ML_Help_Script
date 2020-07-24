'''Basic Data Cleansing Techniques
df – converted pandas DataFrame table object'''

# --- Drop duplicate values ---
df = df.drop_duplicates()

# --- Drop all records (rows) where feature value does not equal to a particular value ---
df = df[df.<feature> != <value>]

# --- Drop columns from the dataset ---
df.drop([<column1>, <column2>], axis = 1, inplace=True)

# --- Display unique values of feature ---
df.<feature>.unique()

# --- Fill missing feature values with (0, 1, string, etc.) ---
df.<feature>.fillna(<value_fill>, inplace = True)

# --- Replace title for another ---
df.replace(<title_to_replace>, <updated_title>, inplace=True)

# --- Replace titles for another --- 
df.replace(<[title1, title2, ...]>, <new_title>, inplace=True)

# --- Display number of missing values by feature (categorical) ---
df.select_dtypes(include=['object']).isnull().sum()

# --- Fill missing categorical values ---
for column in df.select_dtypes(include=['object']):
    df[column] = df[column].fillna('Missing')