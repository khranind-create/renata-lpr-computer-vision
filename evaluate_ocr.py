def levenshtein_distance(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i

    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[len(a)][len(b)]


def character_error_rate(ground_truth, prediction):
    distance = levenshtein_distance(ground_truth, prediction)
    return distance / len(ground_truth)


def word_error_rate(ground_truth, prediction):
    gt_words = ground_truth.split()
    pred_words = prediction.split()
    distance = levenshtein_distance(gt_words, pred_words)
    return distance / len(gt_words)


ground_truth = "M70-O-629"
prediction = "M70-O-629"

cer = character_error_rate(ground_truth, prediction)
wer = word_error_rate(ground_truth, prediction)

print("Stvarna oznaka:", ground_truth)
print("OCR rezultat:", prediction)
print(f"CER: {cer:.2%}")
print(f"WER: {wer:.2%}")