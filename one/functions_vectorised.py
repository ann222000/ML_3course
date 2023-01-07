import numpy as np


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    X_diag = np.diag(X)
    res = np.where(X_diag >= 0)
    return X_diag[res].sum() if np.any(X_diag >= 0) else -1


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    x.sort()
    y.sort()
    return np.array_equal(x, y)



def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    y = np.roll(x, 1)
    res = x[1:] * y[1:]
    return res[np.where(res % 3 == 0)].max() if np.any(res % 3 == 0) else -1


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return np.sum(image * weights, axis=2)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x = x.T
    x = np.repeat(x[0], x[1])
    y = y.T
    y = np.repeat(y[0], y[1])
    return (x * y).sum() if len(x) == len(y) else -1


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    scal_prod = X @ Y.T
    vectors_norm = np.array([np.linalg.norm(X, axis=1)]).T @ np.array([np.linalg.norm(Y, axis=1)])
    idx = np.where(vectors_norm == 0)
    vectors_norm[idx] = 1
    scal_prod[idx] = 1
    return scal_prod / vectors_norm
