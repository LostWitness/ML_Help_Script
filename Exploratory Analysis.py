Exploratory Analysis:

# Dataframe dimensions
df.shape

# Column datatypes
df.dtypes

# Summarize numerical features
df.describe()

# Summarize categorical features
df.describe(include=['object'])

#Display first 5 rows; can speciy number inside to show n rows
df.head()

# Display last 5 rows of data
df.tail()

# Filter and display only df.dtypes that are 'object'
df.dtypes[df.dtypes == 'object']

# Segment by <> and display the means within each class; can do the same with .std()
df.groupby('<>').mean()

# Segment by <> and display the means and standard deviations within each class
df.groupby('<>').agg([np.mean, np.std])

# Loop through categorical feature names and print each one
for feature_names in df.dtypes[df.dtypes == 'object'].index:
    print (feature_names)

# Plot bar plot for each categorical feature
for feature_names in df.dtypes[df.dtypes == 'object'].index:
    sns.countplot(y = feature_names, data=df)
    plt.show()

# Plot histogram grid
df.hist(figsize=(14,14), xrot=-45)
# Clear the text "residue"
plt.show()

# Bar plot for '<insert column name>'
sns.countplot(y = '<>', data=df)

# Boxplot of <> and <>
sns.boxplot(x = '<>', y = '<>', data = df)

# Violinplot of <> and <>
sns.violinplot(y = '<>', x = '<>', data = df)

# Make the figsize 10 x 8
plt.figure(figsize=(9,8))
# Plot heatmap of annotated correlations
sns.heatmap(correlations*100,annot = True ,fmt='.0f', cbar=False)

#For classification problems (bivariate)
sns.lmplot(x='<>', y='<>', hue='<binary target variable>', data=df, fit_reg=False)
# If we want scatter of only one of the target variables
sns.lmplot(x='<>', y='<>', data=df[df.<target column> == '<target value>'], fit_reg=False)

