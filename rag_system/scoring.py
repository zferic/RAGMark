# scoring.py
import collections
import string
import re


def normalize(text):
    text = str(text).lower().strip()
    text = text.rstrip(".")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def exact_match(pred, gold_list):
    pred = normalize(pred)
    gold = {normalize(g) for g in gold_list}
    return int(pred in gold)


def f1_score(pred, gold_list):
    def tokenize(t):
        return normalize(t).split()

    pred_tok = tokenize(pred)
    best_f1 = 0

    for gold in gold_list:
        gold_tok = tokenize(gold)
        common = collections.Counter(pred_tok) & collections.Counter(gold_tok)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        p = num_common / len(pred_tok)
        r = num_common / len(gold_tok)
        f1 = 2 * p * r / (p + r)
        best_f1 = max(best_f1, f1)
    return best_f1


def retrieval_recall(docs, gold_answers):
    """1 if any gold answer appears (substring, normalized) in any retrieved doc, else 0."""
    if not docs:
        return 0
    for ans in gold_answers:
        ans_n = normalize(ans)
        if any(ans_n in normalize(doc) for doc in docs):
            return 1
    return 0


def rouge_l(prediction, ground_truths):
    """
    Compute ROUGE-L score based on the longest common subsequence (LCS).
    """
    def lcs(X, Y):
        """Computes the longest common subsequence (LCS) length."""
        m, n = len(X), len(Y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]

    def tokenize(text):
        return text.lower().strip().split()

    prediction_tokens = tokenize(prediction)
    best_rouge = 0

    for ground_truth in ground_truths:
        ground_truth_tokens = tokenize(ground_truth)
        lcs_length = lcs(prediction_tokens, ground_truth_tokens)

        recall = lcs_length / len(ground_truth_tokens)
        precision = lcs_length / len(prediction_tokens)
        if recall + precision == 0:
            rouge_l = 0
        else:
            rouge_l = (2 * recall * precision) / (recall + precision)

        best_rouge = max(best_rouge, rouge_l)

    return best_rouge
