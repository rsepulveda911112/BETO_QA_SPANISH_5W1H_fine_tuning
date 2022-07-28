import string, re


# def remove_articles(text):
#     regex = re.compile(r"\b(el|la|las|los|lo)\b", re.UNICODE)
#     return re.sub(regex, " ", text)


def white_space_fix(text):
    return " ".join(text.split())


def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def lower(text):
    return text.lower()


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    # white_space_fix(remove_punc(
    return remove_punc(lower(s))


def compute_exact_match(truth, prediction):
    value_1 = normalize_text(prediction)
    value_2 = normalize_text(truth)
    return int(value_1 == value_2)


def compute_exact_match_texts(truths, predictions):
    # return max((compute_exact_match(truth, prediction)) for truth, prediction in enumerate(truths, predictions))
    x = 0
    for truth, prediction in zip(truths, predictions):
        x += compute_exact_match(truth, prediction)
    return x


def compute_f1_texts(truths, predictions):
    # return max((compute_exact_match(truth, prediction)) for truth, prediction in enumerate(truths, predictions))
    x = 0
    for truth, prediction in zip(truths, predictions):
        x += compute_f1(truth, prediction)
    return x/len(truths)


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def get_gold_answers(example):
    """helper function that retrieves all possible true answers from a squad2.0 example"""

    gold_answers = [answer["text"] for answer in example.answers if answer["text"]]

    # if gold_answers doesn't exist it's because this is a negative example -
    # the only correct answer is an empty string
    if not gold_answers:
        gold_answers = [""]

    return gold_answers