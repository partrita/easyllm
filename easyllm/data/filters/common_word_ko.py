from typing import List

from pydantic import BaseModel

COMMON_WORDS_EN = ["the", "be", "to", "of", "and", "that", "have", "with", "this"]
COMMON_WORDS_DE = ["der", "die", "das", "er" "sein", "zu", "ist", "war", "von", "und", "haben", "mit"]


class CommonWordFilter(BaseModel):
    """
    참조: Gopher (Rae et al., 2021)
    설명: 문서에 최소 2개의 일반적인 단어가 포함되어 있는지 확인하고 그렇지 않으면 제거합니다.
    """

    name: str = "common_word"
    common_words: List[str] = COMMON_WORDS_EN
    n: int = 2

    def __call__(self, text):
        words = text.split()
        common_word_counter = 0
        # 일반적인 단어의 수를 계산합니다.
        for word in words:
            if word.lower() in self.common_words:
                common_word_counter += 1
            if common_word_counter >= self.n:
                return False
        # 그렇지 않으면 제거합니다.
        return True
