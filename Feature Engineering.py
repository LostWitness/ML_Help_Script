'''Basic Feature Engineering Techniques
df – converted pandas DataFrame table object'''

# --- Indicator variable for missing <feature>, creating new feature; can do same using boolean masks ---
df[<new_feature>] = df.<feature>.isnull().astype(int)

# --- Convert a feature to an indicator variable ---
df[<feature>] = pd.get_dummies(df.<feature>).<Value in feature that you want to be labelled as 1>