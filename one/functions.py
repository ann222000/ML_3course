from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    s = 0
    is_there = False
    for i in range(len(X)):
        if (len(X[i]) > i) and (X[i][i] >= 0):
            is_there = True
            s += X[i][i]
    return s if is_there else -1



def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    x.sort()
    y.sort()
    return x == y


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    is_there = False
    max_val = 0
    for i in range(len(x) - 1):
        mul_val = x[i] * x[i+1]
        if mul_val % 3 == 0:
            is_there = True
            max_val = max(max_val, mul_val)
    return max_val if is_there else -1


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    result = [[0] * len(image[0]) for i in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[i])):
            for k in range(len(image[i][j])):
                result[i][j] += image[i][j][k] * weights[k]
    return result


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    new_x = [val[0] for val in x for i in range(val[1])]
    new_y = [val[0] for val in y for i in range(val[1])]
    if len(new_x) != len(new_y):
        return -1
    s = 0
    for i in range(len(new_x)):
        s += new_x[i] * new_y[i]
    return s


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    d = len(X[0])
    m = [[0]*len(Y) for i in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            zero1 = True
            zero2 = True
            for k in range(d):
                if X[i][k] != 0:
                    zero1 = False
                if Y[j][k] != 0:
                    zero2 = False
                m[i][j] += X[i][k] * Y[j][k]
            norm_x = sum(map(lambda p: p ** 2, X[i])) ** 0.5
            norm_y = sum(map(lambda p: p ** 2, Y[j])) ** 0.5
            m[i][j] = m[i][j] / norm_x / norm_y if not zero1 and not zero2 else 1
    return m
