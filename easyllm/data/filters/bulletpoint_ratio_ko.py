from typing import List

from pydantic import BaseModel


class BulletpointRatioFilter(BaseModel):
    """
    참조: Gopher (Rae et al., 2021)
    설명: 문서의 90% 이상이 글머리 기호이면 제거합니다.
    """

    name: str = "bulletpoint_ratio"
    potential_bullet_points: List[str] = [
        "•",
        "‣",
        "⁃",
        "⁌",
        "⁍",
        "∙",
        "○",
        "●",
        "◘",
        "◦",
        "⦾",
        "⦿",
        "-",
    ]
    remove_percentage: float = 0.9

    def __call__(self, text):
        # 텍스트를 줄로 분할합니다.
        lines = text.split("\n")
        num_bullet_points = 0
        for line in lines:
            # 줄이 글머리 기호인지 확인합니다.
            if line.startswith(tuple(self.potential_bullet_points)):
                num_bullet_points += 1
        # 글머리 기호 대 줄의 비율이 제거 비율보다 큰지 확인합니다.
        if num_bullet_points / len(lines) > self.remove_percentage:
            return True
        # 그렇지 않으면 유지합니다.
        return False
