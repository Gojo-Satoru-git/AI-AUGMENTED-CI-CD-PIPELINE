"""Tiny block vocabulary shared by the chapter files.

Inline markup inside text: **bold**, *italic*.
Cross references: {T:key} -> "Table 5.3", {F:key} -> "Figure 3.1", {A:key} -> "Algorithm 4.1",
citations: {R:key} -> "[7]" (numbered by first citation).
"""


def CH(title):
    return ("chapter", title)


def H2(title):
    return ("h2", title)


def H3(title):
    return ("h3", title)


def P(text):
    return ("p", text)


def BUL(*items):
    return ("bul", list(items))


def TAB(key, caption, header, rows, widths):
    """widths in cm, must sum to <= 16.1"""
    return ("table", key, caption, header, rows, widths)


def FIG(key, caption, file, width_cm):
    return ("fig", key, caption, file, width_cm)


def ALG(key, caption, code):
    return ("alg", key, caption, code)
