import string

def parse_cat(cat_str):
    return list(cat_str)[0][0]

def strip_punctuation(input):
    return input.translate(str.maketrans('', '', string.punctuation))

def lower_case(input):
    return input.lower()

def strip_spaces(input):
    return "".join(input.split())