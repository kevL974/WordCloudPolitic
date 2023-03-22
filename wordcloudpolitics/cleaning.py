import re


def to_lower_case(text: str) -> str:
    """
    The function transforms strings in lower case
    :param text: the text that you want to apply lower case function.
    :return: text in lower case
    """
    assert text is not None
    return text.lower()


def remove_url(text: str) -> str:
    """
    The function removes all URLs inside the input text.
    :param text: text with URLs.
    :return: text without URLs.
    """
    assert text is not None
    return re.sub(
        "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)",
        "", text)


def transform_accented_character(text: str) -> str:
    """
    The function replaces all accented characters by their basic character, example :é -> e
    :param text: text with accented characters.
    :return: text withour accented characters.
    """
    assert text is not None
    clean = text.replace("à", "a")
    clean = clean.replace("é", "e")
    clean = clean.replace("è", "e")
    clean = clean.replace("ê", "e")
    clean = clean.replace("ë", "e")
    clean = clean.replace("î", "i")
    clean = clean.replace("ï", "i")
    clean = clean.replace("ô", "o")
    clean = clean.replace("ù", "u")
    return clean


def remove_special_character(text: str) -> str:
    """
    The function removes all special characters inside the input text.er.
    :param text: text with special characters.
    :return: text without special characters.
    """
    assert text is not None
    return re.sub(r'[^\w\s]', '', text)


def remove_multispaces(text: str) -> str:
    """
    The function removes multi-spaces from input text.
    :param text: text with multi-spaces.
    :return: text withour multi-spaces.
    """
    assert text is not None
    return re.sub(r'\s+', ' ', text)
