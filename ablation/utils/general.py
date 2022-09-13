from sklearn.utils import resample


def sample(X, nsamples=100, random_state=0):
    if nsamples >= X.shape[0]:
        return X
    else:
        return resample(X, n_samples=nsamples, random_state=random_state)
