import numpy as np
from scipy import optimize


class BinaryEstimatorSVM:


    def __init__(self, lr=0.01, C=1.0, n_epochs=100, batch_size=16, fit_intercept=True, drop_last=True):


        self.lr = lr
        self.C = C
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.drop_last = drop_last
        self.coef_ = None
        self.intercept_ = None
        self.n_classes_ = None


    def predict(self, X):

        X = np.asarray(X, dtype=float)
        b = float(np.asarray(self.intercept_).reshape(-1)[0]) if self.fit_intercept else 0.0
        return X @ self.coef_ + b


    def loss(self, X, y_true):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y_true, dtype=float).reshape(-1)
        w = self.coef_.reshape(-1)
        b = float(np.asarray(self.intercept_).reshape(-1)[0]) if self.fit_intercept else 0.0

        decision = X @ w + b
        margin = y * decision
        hinge = np.maximum(0.0, 1.0 - margin)

        # Усредненим
        return 0.5 * np.sum(w ** 2) + self.C * np.mean(hinge)
      
    def loss_grad(self, X, y_true):


        X = np.asarray(X, dtype=float)
        y = np.asarray(y_true, dtype=float).reshape(-1)
        w = self.coef_.reshape(-1)
        b = float(np.asarray(self.intercept_).reshape(-1)[0]) if self.fit_intercept else 0.0

        decision = X @ w + b
        margin = y * decision
        mask = (margin < 1.0).astype(float)
        n = X.shape[0]

        grad_w = w - self.C * (X.T @ (y * mask))
        grad_w = grad_w.reshape(self.coef_.shape)

        if self.fit_intercept:
            grad_b = -np.sum(y * mask)
            grad_intercept = np.asarray(grad_b)
        else:
            grad_intercept = None

        return grad_w, grad_intercept

    def step(self, grad, grad_intercept):


        self.coef_ = self.coef_ - self.lr * grad
        if self.fit_intercept and grad_intercept is not None:
            self.intercept_ = self.intercept_ - self.lr * grad_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape

        self.n_classes_ = 2
        self.coef_ = np.zeros((n_features, 1))
        self.intercept_ = np.zeros(1) if self.fit_intercept else None

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            batches = [indices[i:i + self.batch_size] for i in range(0, n_samples, self.batch_size)]

            if self.drop_last:
                batches = [b for b in batches if len(b) == self.batch_size]
                if len(batches) == 0:
                    batches = [indices]

            for batch in batches:
                grad, grad_intercept = self.loss_grad(X[batch], y[batch])

                # Прямой шаг градиентного спуска.
                # Никаких делений, так как loss_grad уже возвращает нужную формулу.
                self.step(grad, grad_intercept)

        return self


def one_vs_rest(y, n_classes=None):


    y = np.asarray(y).reshape(-1)
    if n_classes is None:
        n_classes = int(np.max(y)) + 1
    Y = np.full((len(y), n_classes), -1.0)
    for c in range(n_classes):
        Y[y == c, c] = 1.0
    return Y


class LinearPrimalSVM:


    def __init__(self, lr=0.01, C=1.0, n_epochs=100, batch_size=16, fit_intercept=True, drop_last=True):


        self.lr = lr
        self.C = C
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.drop_last = drop_last
        self.n_classes_ = None
        self.list_of_models = []


    def predict(self, X):


        X = np.asarray(X, dtype=float)
        decisions = np.column_stack([model.predict(X) for model in self.list_of_models])
        return np.argmax(decisions, axis=1)

    def fit(self, X, y):


        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        self.n_classes_ = int(np.max(y)) + 1
        self.list_of_models = []
        Y = one_vs_rest(y, self.n_classes_)

        for c in range(self.n_classes_):
            model = BinaryEstimatorSVM(
                lr=self.lr, C=self.C, n_epochs=self.n_epochs,
                batch_size=self.batch_size, fit_intercept=self.fit_intercept,
                drop_last=self.drop_last
            )
            model.fit(X, Y[:, c])
            self.list_of_models.append(model)
        return self
    
    
def kernel_linear(x1, x2):

    return np.array(np.dot(np.asarray(x1), np.asarray(x2)))


def kernel_poly(x1, x2, d=2):


    x1, x2 = np.asarray(x1), np.asarray(x2)
    return np.array((np.dot(x1, x2) + 1.0) ** d)


def kernel_rbf(x1, x2, l=1.0):
    x1, x2 = np.asarray(x1), np.asarray(x2)
    dist2 = np.sum((x1 - x2) ** 2)
    return np.array(np.exp(-dist2 / (2.0 * l ** 2)))

def lagrange(gramm_matrix, alpha):

    alpha = np.asarray(alpha).reshape(-1)
    return np.array(np.sum(alpha) - 0.5 * alpha @ gramm_matrix @ alpha)

def lagrange_derive(gramm_matrix, alpha):


    alpha = np.asarray(alpha).reshape(-1)
    return np.ones_like(alpha) - gramm_matrix @ alpha

def one_vs_one(X, y, n_classes=None):


    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y) if n_classes is None else np.arange(n_classes)
    result = []

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            class_plus, class_minus = classes[i], classes[j]
            mask = (y == class_plus) | (y == class_minus)
            X_cut = X[mask]
            y_cut = np.where(y[mask] == class_plus, 1, -1)
            result.append((X_cut, y_cut, class_plus, class_minus))
    return result


class SoftMarginSVM:

    
    def __init__(self, C, kernel_func, classes_names=None):


        self.C = C                                 
        self.alpha = None
        self.supportVectors = None
        self.supportLabels = None
        self.supportalpha = None
        self.kernel = kernel_func
        self.classes_names = classes_names
        self.b = None

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        if self.supportVectors is None or len(self.supportVectors) == 0:
            decision = np.full((n, 1), 0.0 if self.b is None else self.b)
        else:
            K = np.zeros((len(self.supportVectors), n))
            for i, sv in enumerate(self.supportVectors):
                for j, x in enumerate(X):
                    K[i, j] = self.kernel(sv, x)
            weights = self.supportalpha * self.supportLabels
            decision = (weights @ K + self.b).reshape(-1, 1)

        pred = np.where(decision >= 0, 1, -1)
        if self.classes_names is None:
            return pred.astype(float)

        # ВАЖНО: для positive (pred == 1) берем [0], для negative (pred == -1) берем [1]
        return np.where(pred == 1, self.classes_names[0], self.classes_names[1]).astype(float)


    def fit(self, X, y):


        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        n = X.shape[0]

        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j])

        G = K * np.outer(y, y)
        alpha0 = np.zeros(n)
        bounds = [(0.0, self.C) for _ in range(n)]
        constraint = {'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y}

        res = optimize.minimize(
            fun=lambda a: -lagrange(G, a),
            x0=alpha0,
            jac=lambda a: -lagrange_derive(G, a),
            bounds=bounds,
            constraints=constraint,
            method='SLSQP',
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        alpha = np.clip(res.x, 0.0, self.C)
        self.alpha = alpha
        eps = 1e-5
        sv_mask = (alpha > eps) & (alpha <= self.C + eps)

        self.supportVectors = X[sv_mask]
        self.supportLabels = y[sv_mask]
        self.supportalpha = alpha[sv_mask]

        decision_values = K @ (alpha * y)
        free_mask = (alpha > eps) & (alpha < self.C - eps)

        if np.any(free_mask):
            self.b = float(np.mean(y[free_mask] - decision_values[free_mask]))
        elif np.any(sv_mask):
            self.b = float(np.mean(y[sv_mask] - decision_values[sv_mask]))
        else:
            self.b = 0.0
        return self


class NonLinearDualSVM:


    def __init__(self, C=1.0, kernel='rbf', kernel_parameter=1.0):

        self.C = C

        if kernel == 'poly':
          self.kernel = lambda x, y: kernel_poly(x, y, d=kernel_parameter)
        elif kernel == 'rbf':
          self.kernel = lambda x, y: kernel_rbf(x, y, l=kernel_parameter)
        else:
          self.kernel = kernel_linear

        self.kernel.__name__='kernel'

        self.estimators = []


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        self.estimators = []

        # ВАЖНО: передаем [class_plus, class_minus], чтобы соответствовать новой логике predict
        for X_cut, y_cut, class_plus, class_minus in one_vs_one(X, y):
            model = SoftMarginSVM(C=self.C, kernel_func=self.kernel, classes_names=[class_plus, class_minus])
            model.fit(X_cut, y_cut)
            self.estimators.append(model)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if len(self.estimators) == 0:
            return np.zeros(len(X), dtype=int)

        votes = np.column_stack([model.predict(X) for model in self.estimators])

        predictions = []
        for row in votes:
            values, counts = np.unique(row, return_counts=True)
            predictions.append(values[np.argmax(counts)])
        return np.asarray(predictions).astype(int)
