import string

def parse_cat(cat):
    '''
    Parse the category from a dataset entry

    @param cat: Listlike containing hierarchical classes for a product
    '''
    return list(cat)[0][0]

def strip_punctuation(input):
    '''
    Removes all punctuation from a string

    @param input: String to be processed
    '''
    return input.translate(str.maketrans('', '', string.punctuation))

def lower_case(input):
    '''
    Makes all letters in a string lowercase

    @param input: String to be processed
    '''
    return input.lower()

def strip_spaces(input):
    '''
    Removes all spaces from a string

    @param input: String to be processed
    '''
    return "".join(input.split())