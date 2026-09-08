from evaluation.metrics import batch_f1_score, f1_score, normalize_answer


def test_normalize_answer_removes_punctuation_and_numeric_commas():
    assert normalize_answer("The, Value: 1,024!") == "the value 1024"


def test_f1_uses_token_multiplicity():
    assert f1_score("blue blue red", "blue red red") == 2 / 3


def test_batch_accepts_multiple_references():
    assert batch_f1_score(["Paris"], [["London", "Paris"]]) == 1.0
