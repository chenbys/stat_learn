import matplotlib.pyplot as plt


def by_pca(data, n_components):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit(data).transform(data)
    print(f'pca ratio: {pca.explained_variance_ratio_.sum()}')
    return reduced, pca.explained_variance_ratio_.sum()


def by_kernel_pca(data, n_components):
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components=n_components)
    reduced = kpca.fit_transform(data)
    # print(kpca.explained_variance_ratio_.sum())
    return reduced, 1


def normlize(data, n_components):
    from sklearn import preprocessing
    normalizer = preprocessing.Normalizer().fit(data)
    normalized = normalizer.transform(data)
    return normalized, n_components


def row(data, n_components):
    return data, n_components
