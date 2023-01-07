import re


def find_shortest(l):
    a = re.findall('[a-zA-Z]+', l)
    return min(len(i) for i in a) if a else 0
