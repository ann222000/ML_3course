def check(x: str, file: str):
    word_list = x.lower().split(' ')
    word_set = set(word_list)
    word_amount = {word: word_list.count(word) for word in word_set}
    word_amount = sorted(word_amount.items())
    with open(file, 'w') as f:
        for word in word_amount:
            f.write(f"{word[0]} {word[1]}\n")
