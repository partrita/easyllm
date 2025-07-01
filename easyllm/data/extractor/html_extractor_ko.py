#
from inscriptis import get_text
from inscriptis.css_profiles import CSS_PROFILES
from inscriptis.model.config import ParserConfig
from pydantic import BaseModel
from readability import Document

INSCRIPTIS_CONFIG = ParserConfig(css=CSS_PROFILES["strict"])


class HtmlExtractor(BaseModel):
    """
    설명: mozzilas readability 및 inscriptis를 사용하여 HTML 문서에서 텍스트를 추출합니다.
    """

    name: str = "html_extractor"
    min_doc_length: int = 25

    def __call__(self, document: str) -> str:
        parsed_doc = Document(document, min_text_length=self.min_doc_length)
        clean_html = parsed_doc.summary(html_partial=True)
        content = get_text(clean_html, INSCRIPTIS_CONFIG).strip()
        return content
