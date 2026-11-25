with open("data/input.txt", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines()
print(len(lines))

print(lines[0])
print(lines[-3:])
print(lines[::10000])

words = text.split()
print(len(words))
print(len(set(words)))

chars = set(text)
chars = sorted(chars)
print(len(chars))
chars_list = list(chars)
print(chars_list[0:5])
print(chars_list[-5:])

chars_str = "".join(chars)
print(chars_str)

print("z" in chars)
print("7" in chars)
print(chars_list.index("a"))
print(chars_list.index("A"))