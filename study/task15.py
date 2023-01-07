from typing import List


def hello(name: str = None) -> str:
    if (name is None) or (name == ""):
        return "Hello!"
    else:
        return f"Hello, {name}!"


def int_to_roman(num: int) -> str:
    ones = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    tens = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
    hundreds = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
    thousands = ["", "M", "MM", "MMM"]
    return thousands[num // 1000] + hundreds[num // 100 % 10] + tens[num // 10 % 10] + ones[num % 10]


def longest_common_prefix(strs_input: List[str]) -> str:
    if len(strs_input) == 0:
        return ""
    begin_set = set()
    step = 0
    while ((len(begin_set) == 1) or (step == 0)) and (len(strs_input[0]) >= step):
        step += 1
        begin_set = set(str_input.strip()[:step] for str_input in strs_input if len(str_input.strip()) > 0)
    return (begin_set.pop())[:step - 1] if len(begin_set) > 0 else ""


def is_prime(number: int) -> bool:
    if number <= 1:
        return False
    i = 2
    while i <= round(number ** 0.5):
        if number % i == 0:
            return False
        i += 1
    return True


def primes() -> int:
    number = 1
    while True:
        number += 1
        if is_prime(number):
            yield number


class BankCard:
    def __init__(self, total_sum: int, balance_limit: int = None):
        self.total_sum = total_sum
        self.balance_limit = balance_limit

    def __str__(self):
        return "To learn the balance call balance."

    def __call__(self, sum_spent: int):
        if sum_spent > self.total_sum:
            print("Can't spend sum_spent dollars")
            raise ValueError()
        else:
            self.total_sum -= sum_spent
            print("You spent sum_spent dollars")

    @property
    def balance(self) -> int:
        if self.balance_limit is None:
            return self.total_sum
        self.balance_limit -= 1
        if self.balance_limit < 0:
            print("Balance check limits exceeded.")
            raise ValueError()
        return self.total_sum

    def put(self, some_put: int):
        self.total_sum += some_put
        print("You put sum_put dollars.")

    def __add__(self, another: 'BankCard') -> 'BankCard':
        return BankCard(self.total_sum + another.total_sum, max(self.balance_limit, another.balance_limit))
