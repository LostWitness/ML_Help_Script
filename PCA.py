'''Principal Component Analysis'''

from sklearn.decomposition import PCA

# --- Load instance of PCA transformation ---
pca = PCA()

# --- Load 1 component ---
pca = PCA(n_components=1)

# --- Fit the instance ---
pca.fit(<data>)

# --- Display principal components ---
pca.components_

# --- Generate new features ---
PC = pca.transform(X_scaled)

# --- Display explained variance ratio ---
pca.explained_variance_ratio_

# --- Cumulative explained variance ---
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# --- Plot cumulative explained variance ---
plt.plot(range(len(cumulative_explained_variance)), cumulative_explained_variance)