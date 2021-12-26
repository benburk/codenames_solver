"""Codenames clue finder using glove vectors.

We can express the Codenames problem as taking a set of "target" words and a set of
"bad" words, then trying to find candidate words that are close to the targets and far
from the bad words.

Links
- https://jsomers.net/glove-codenames/
"""
from operator import itemgetter

import numpy as np
import numpy.typing as npt


def load_embeddings(filename: str) -> dict[str, npt.NDArray[np.float64]]:
    """Load glove word embeddings from file.

    Download glove vectors from https://nlp.stanford.edu/projects/glove/
    $ wget http://nlp.stanford.edu/data/glove.42B.300d.zip
    $ unzip glove.42B.300d.zip
    $ head -n 50000 glove.42B.300d.txt > top_50000.txt

    :param filename: The name of the file
    :return: Mapping of word to vector
    """
    embeddings = {}
    with open(filename, encoding="utf8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings[word] = vector
    return embeddings


EMBEDDINGS = load_embeddings("/Users/benburk/Desktop/top_50000.txt")


def distance(word: str, reference: str) -> float:
    """Cosine similarity between words.
    https://en.wikipedia.org/wiki/Cosine_similarity

    :param word: The word to compare
    :param reference: The reference word to compare with
    :return: The cosine similarity score of the two words
    """
    vec1 = EMBEDDINGS[word]
    vec2 = EMBEDDINGS[reference]

    dot_product = np.dot(vec1, vec2)  # type: ignore[no-untyped-call]
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)  # type: ignore[no-untyped-call]
    cos_sim: float = dot_product / magnitude
    return 1 - cos_sim


def score_clue(clue: str, good_words: list[str], bad_words: list[str]) -> float:
    """Score how good a clue is.

    For a given clue, the score is the sum of distances from the bad words minus the
    sum of distances from the good words. When the score is larger, the clue is better.
    The constant 4.0 expresses that closeness to the good words is more important than
    farness from the bad words.

    :param clue: The candidate clue word
    :param good_words: words the clue should be similar to
    :param bad_words: words the clue should be dissimilar to
    :return: The score of the clue
    """
    if clue in good_words + bad_words:
        return -999  # clue can't be one of the good words or bad words
    return sum((distance(clue, bad_word) for bad_word in bad_words)) - 4.0 * sum(
        (distance(clue, good_word) for good_word in good_words)
    )


def minimax(clue: str, good_words: list[str], bad_words: list[str]) -> float:
    """Minimax.

    :param clue: The candidate clue word
    :param good_words: words the clue should be similar to
    :param bad_words: words the clue should be dissimilar to
    :return: The score of the clue
    """
    if clue in good_words + bad_words:
        return -999
    return min((distance(clue, b) for b in bad_words)) - max(
        (distance(clue, a) for a in good_words)
    )


def get_best_clues(
    good_words: list[str], bad_words: list[str], size: int = 50
) -> list[tuple[str, float]]:
    """Find the best clue for a list of target words and words to avoid.

    :param good_words: The list of target words
    :param bad_words: The list of words to avoid
    :param size: The number of clues to return
    :return: The clues and their associated scores
    """
    candidates = sorted(
        EMBEDDINGS,
        key=lambda word: score_clue(word, good_words, bad_words),
        reverse=True,
    )[:250]
    best = sorted(
        ((word, minimax(word, good_words, bad_words)) for word in candidates),
        key=itemgetter(1),  # sort by second element
        reverse=True,
    )[:size]

    return best


def main() -> None:
    """main method"""
    # good_words = ["iron", "ham", "beijing"]
    # bad_words = ["fall", "witch", "note", "cat", "bear", "ambulance"]
    good_words = ["square", "shallow"]
    bad_words = ["spike", "balloon", "plan", "portfolio", "handle"]

    good_words = ["rose", "vision"]
    bad_words = ["vest", "cone"]

    # good_words = ["church", "cat", "atlantis"]
    # bad_words = ["fair", "eye", "aztec", "buck", "pin", "hospital"]
    results = get_best_clues(good_words, bad_words)
    for i, (word, score) in enumerate(results):
        print(f"{i}: {word} ({score})")


if __name__ == "__main__":
    main()
