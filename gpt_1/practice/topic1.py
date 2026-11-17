with open("data/input.txt", encoding="utf-8") as f:
    text = f.read()

num_chars = len(text)
print(num_chars)

first_13_chars = text[:13]
print(first_13_chars)
first_15_59_chars = text[15:60]
print(first_15_59_chars)

print(text.count("Juliet"))
print(text.count("JULIET"))
print(text.count("love"))

romeo_index = text.find("ROMEO")
print(romeo_index)
print(text[romeo_index:romeo_index+6])


if text.find("Hamlet") != -1:
    print("yes")
else:
    print("no")

print(text.count("the"))
print(text.lower().count("the"))


print(text[10:20])
print(text.count("\n"))


num_pages = num_chars // 3000
chars_left = num_chars % 3000
print(num_pages)
print(chars_left)

num_e = text.count("e")
percentage_e = num_e / num_chars * 100
percentage_e = round(percentage_e, 2)
print(percentage_e,"%")

print(f'The book has {num_chars} characters and the word \"love\" appears {text.count("love")} times.')