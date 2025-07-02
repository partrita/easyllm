import re

from pydantic import BaseModel

policy_substrings = [
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "privacy overview",
    "use of cookies",
    "use cookies",
    "privacy & cookies policy",
    "privacy and cookies policy",
    "This website uses cookies to improve your experience while you "
    "navigate through the website. Out of these cookies, the cookies "
    "that are categorized as necessary are stored on your browser as they "
    "are essential for the working of basic functionalities of the website. "
    "We also use third-party cookies that help us analyze and understand how "
    "you use this website. These cookies will be stored in your browser only "
    "with your consent. You also have the option to opt-out of these "
    "cookies. But opting out of some of these cookies may have an effect "
    "on your browsing experience.".lower(),
    "Necessary cookies are absolutely essential for the website to "
    "function properly. This category only includes cookies that "
    "ensures basic functionalities and security features of the website. "
    "These cookies do not store any personal information.".lower(),
    "Any cookies that may not be particularly necessary for the website "
    "to function and is used specifically to collect user personal data "
    "via analytics, ads, other embedded contents are termed as non-necessary "
    "cookies. It is mandatory to procure user consent prior to running these "
    "cookies on your website.".lower(),
    "This site uses cookies, including for analytics, personalization, and "
    "advertising purposes. For more information or to change your "
    "cookie settings, click here.".lower(),
    "If you continue to browse this site without changing your cookie "
    "settings, you agree to this use. AcceptRead More".lower(),
]


class CookieBannerFilter(BaseModel):
    """
    참조: C4 Raffel et al.
    설명: 문서의 40% 이상에 쿠키, 서비스 약관, 개인 정보 보호 정책 등에 대한 용어가 포함되어 있으면 문서를 제거합니다. 외부 목록이 필요합니다.
    """

    name: str = "cookie_banner"
    regex: re.Pattern = re.compile(r"(terms of use|privacy policy|copyright|all rights reserved)", re.IGNORECASE)
    remove_percentage: float = 0.4

    def __call__(self, text):
        # 정규식이 일치하는지 확인합니다.
        raise NotImplementedError("CookieBannerFilter가 아직 구현되지 않았습니다.")
