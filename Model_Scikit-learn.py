''' Data Modeling Techniques using sklearn
df – converted pandas DataFrame table object'''

# --- Create new object for target variable ---
y = df.<feature>

# --- Create new object for input features ---
X = df.drop('<target_feature>', axis = 1)

# --- Variable to split data into train and test data ---
train_test_split

# --- Create model Pipeline ---
from sklearn.pipeline import make_pipeline
# --- Standardization ---
from sklearn.preprocessing import StandardScaler

# --- ML 1: a dictionary holding different algorithms ---
pipelines = {
	'lasso': make_pipeline(StandardScaler(), lasso()),
	...
}

# --- ML 2: hyperparameters grid – one per algorithm ---
lasso_hyperparameters = { 
    'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10] 
}

# --- ML 3: hyperparameters dictionary --- 
hyperparameters = {
    'lasso': lasso_hyperparameters,
    ...
}

# --- List tuneable hyperparameters of our Lasso pipeline ---
pipelines['lasso'].get_params()


''' GridSearch techniques '''
# --- GridSearch approach for cross-validation ---
from sklearn.model_selection import GridSearchCV
# --- Create cross-validation object from Lasso pipeline and Lasso hyperparameters ---
model = GridSearchCV(pipelines['lasso'], hyperparameters['lasso'], cv=10, n_jobs=-1)

# --- Loop through all models and fit models ---
	# --- Create empty dictionary for storing fitted models ---
	fitted_models = {}

	# --- Loop through model pipelines, tuning each one and saving it to fitted_models ---
	for name, pipeline in pipelines.items():
	    # --- Create cross-validation object from pipeline and hyperparameters ---
	    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
	    
	    # --- Fit model on X_train, y_train ---
	    model.fit(X_train, y_train)
	    
	    # --- Store model in fitted_models[name] ---
	    fitted_models[name] = model
	    
	    # --- Display the result ---
	    print(name, 'has been fitted.')

# --- Display best score ---
model.best_score_
# --- Display best parameters for a given model ---
model.best_estimator_

# --- Import r2_score and mean_absolute_error functions ---
from sklearn.metrics import r2_score, mean_absolute_error

# --- Predict test set using fitted Random Forest ---
pred = fitted_models['rf'].predict(X_test)

# --- Calculate and display R^2 and MAE ---
print( 'R^2:', r2_score(y_test, pred ))
print( 'MAE:', mean_absolute_error(y_test, pred))




''' Classification techniques '''

# --- Display probabilities of prediction ---
model.predict_proba

# --- Import Classification metrics ROC, AUC ---
from sklearn.metrics import roc_curve, auc

def fit_and_plot_classifier(clf):

    clf.fit(X, y)
    
    # --- Predict and take second value of each prediction ---
    pred = clf.predict_proba(X)
    pred = [p[1] for p in pred]
    
    plt.scatter(X, y)
    plt.plot(X, pred, 'k--')
    plt.show()
    
    # --- Return fitted model and predictions ---
    return clf, pred

# --- Calculate ROC curve ---
fpr, tpr, thresholds = roc_curve(y_test, pred)
# --- Calculate AUROC ---
print( auc(fpr, tpr) )



''' Cluster Analysis techniques '''

# --- Scikit-Learn KMeans algorithm ---
from sklearn.cluster import KMeans

# --- K-Means model pipeline ---
k_means = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=123))

# --- Fit K-Means pipeline ---
k_means.fit(base_df)

# --- Save clusters to base_df ---
base_df['cluster'] = k_means.predict(base_df)

# --- Scatterplot, rendered by cluster ---
sns.lmplot(x='total_sales', y='avg_cart_value', hue='cluster', data=base_df, fit_reg=False)

# --- Adjusted Rand index ---
from sklearn.metrics import adjusted_rand_score