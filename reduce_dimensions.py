import matplotlib.pyplot as plt


def by_pca(data, n_components):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit(data).transform(data)
    print(pca.explained_variance_ratio_.sum())
    return reduced, pca.explained_variance_ratio_.sum()
