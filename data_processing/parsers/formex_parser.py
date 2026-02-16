"""
FORMEX XML Parser for EUR-Lex Legal Documents

Parses FORMEX (Formalized Exchange of Electronic Publications) XML format
used by EUR-Lex for official EU legal documents.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from lxml import etree
from dataclasses import dataclass, asdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Article:
    """Represents a single article in a legal document"""
    article_id: str
    title: Optional[str]
    text: str
    level: Optional[str]
    paragraphs: List[str]


@dataclass
class DocumentMetadata:
    """Metadata for a legal document"""
    celex: Optional[str]
    language: str
    doc_type: Optional[str]
    date: Optional[str]
    title: Optional[str]
    subject_matter: List[str]


@dataclass
class ParsedDocument:
    """Complete parsed FORMEX document"""
    metadata: DocumentMetadata
    articles: List[Article]
    full_text: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'metadata': asdict(self.metadata),
            'articles': [asdict(art) for art in self.articles],
            'full_text': self.full_text
        }


class FORMEXParser:
    """Parser for FORMEX XML documents from EUR-Lex"""

    # Namespace handling for FORMEX
    NAMESPACES = {
        'formex': 'http://formex.publications.europa.eu/schema/formex-05.56-20160701.xd',
        'html': 'http://www.w3.org/1999/xhtml'
    }

    def __init__(self, remove_formatting: bool = True, max_text_length: Optional[int] = None):
        """
        Initialize FORMEX parser

        Args:
            remove_formatting: Remove XML formatting tags from text
            max_text_length: Maximum text length for truncation (None = no limit)
        """
        self.remove_formatting = remove_formatting
        self.max_text_length = max_text_length

    def parse_file(self, xml_path: Union[str, Path]) -> Optional[ParsedDocument]:
        """
        Parse a FORMEX XML file

        Args:
            xml_path: Path to XML file

        Returns:
            ParsedDocument or None if parsing fails
        """
        xml_path = Path(xml_path)

        if not xml_path.exists():
            logger.error(f"File not found: {xml_path}")
            return None

        try:
            # Parse XML with recovery mode for malformed XML
            parser = etree.XMLParser(recover=True, remove_blank_text=True)
            tree = etree.parse(str(xml_path), parser)

            # Extract metadata
            metadata = self._extract_metadata(tree)

            # Extract articles
            articles = self._extract_articles(tree)

            # Extract full text
            full_text = self._extract_full_text(tree)

            return ParsedDocument(
                metadata=metadata,
                articles=articles,
                full_text=full_text
            )

        except etree.XMLSyntaxError as e:
            logger.error(f"XML syntax error in {xml_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing {xml_path}: {e}")
            return None

    def _extract_metadata(self, tree: etree._ElementTree) -> DocumentMetadata:
        """Extract document metadata"""
        root = tree.getroot()

        # Extract CELEX number
        celex = self._find_celex(root)

        # Extract language
        language = self._find_language(root)

        # Extract document type
        doc_type = self._find_doc_type(root)

        # Extract date
        date = self._find_date(root)

        # Extract title
        title = self._find_title(root)

        # Extract subject matter
        subjects = self._find_subjects(root)

        return DocumentMetadata(
            celex=celex,
            language=language,
            doc_type=doc_type,
            date=date,
            title=title,
            subject_matter=subjects
        )

    def _find_celex(self, root: etree._Element) -> Optional[str]:
        """Extract CELEX number"""
        # Try multiple possible locations
        patterns = [
            ".//IDENTIFIER.CELEX",
            ".//CELEX",
            ".//*[@IDENTIFIER.CELEX]",
            ".//*[contains(@*, 'CELEX')]"
        ]

        for pattern in patterns:
            elements = root.xpath(pattern)
            if elements:
                text = self._extract_text(elements[0])
                if text:
                    # Clean CELEX number
                    celex = re.sub(r'\s+', '', text)
                    return celex

        return None

    def _find_language(self, root: etree._Element) -> str:
        """Extract language code"""
        # Try xml:lang attribute
        lang = root.get('{http://www.w3.org/XML/1998/namespace}lang')
        if lang:
            return lang.lower()

        # Try LANG attribute
        lang = root.get('LANG')
        if lang:
            return lang.lower()

        # Try to find in document structure
        elements = root.xpath(".//*[@LANG]")
        if elements:
            lang = elements[0].get('LANG')
            if lang:
                return lang.lower()

        # Default to unknown
        return 'unknown'

    def _find_doc_type(self, root: etree._Element) -> Optional[str]:
        """Extract document type (regulation, directive, etc.)"""
        # Try to find in title or metadata
        title_elem = root.find(".//TITLE")
        if title_elem is not None:
            title_text = self._extract_text(title_elem).upper()

            # Check for common document types
            if 'REGULATION' in title_text:
                return 'regulation'
            elif 'DIRECTIVE' in title_text:
                return 'directive'
            elif 'DECISION' in title_text:
                return 'decision'
            elif 'RECOMMENDATION' in title_text:
                return 'recommendation'
            elif 'OPINION' in title_text:
                return 'opinion'

        return None

    def _find_date(self, root: etree._Element) -> Optional[str]:
        """Extract document date"""
        # Try multiple date fields
        date_patterns = [
            ".//DATE",
            ".//DATE.DOCUMENT",
            ".//DATE.ADOPTION",
            ".//*[@DATE]"
        ]

        for pattern in date_patterns:
            elements = root.xpath(pattern)
            if elements:
                date_text = self._extract_text(elements[0])
                if date_text:
                    return date_text

        return None

    def _find_title(self, root: etree._Element) -> Optional[str]:
        """Extract document title"""
        title_elem = root.find(".//TITLE")
        if title_elem is not None:
            return self._extract_text(title_elem)

        return None

    def _find_subjects(self, root: etree._Element) -> List[str]:
        """Extract subject matter classifications"""
        subjects = []

        # Try to find subject elements
        subject_patterns = [
            ".//SUBJECT.MATTER",
            ".//DESCRIPTOR",
            ".//EUROVOC"
        ]

        for pattern in subject_patterns:
            elements = root.xpath(pattern)
            for elem in elements:
                subject = self._extract_text(elem)
                if subject:
                    subjects.append(subject)

        return subjects

    def _extract_articles(self, tree: etree._ElementTree) -> List[Article]:
        """Extract all articles from document"""
        root = tree.getroot()
        articles = []

        # Find all article elements
        article_elements = root.xpath(".//ARTICLE | .//ART")

        for idx, art_elem in enumerate(article_elements):
            article_id = art_elem.get('ID') or art_elem.get('NUM') or f"article_{idx+1}"
            level = art_elem.get('LEVEL')

            # Extract article title
            title_elem = art_elem.find(".//TI.ART | .//TITLE")
            title = self._extract_text(title_elem) if title_elem is not None else None

            # Extract paragraphs
            paragraphs = []
            para_elements = art_elem.xpath(".//PARAG | .//P | .//AL")
            for para_elem in para_elements:
                para_text = self._extract_text(para_elem)
                if para_text:
                    paragraphs.append(para_text)

            # Extract full article text
            text = self._extract_text(art_elem)

            if text:  # Only add if there's actual text
                articles.append(Article(
                    article_id=article_id,
                    title=title,
                    text=text,
                    level=level,
                    paragraphs=paragraphs
                ))

        return articles

    def _extract_full_text(self, tree: etree._ElementTree) -> str:
        """Extract full document text"""
        root = tree.getroot()

        # Try to find main text body
        body_elem = root.find(".//BODY | .//TEXT")
        if body_elem is not None:
            text = self._extract_text(body_elem)
        else:
            # Fall back to extracting from root
            text = self._extract_text(root)

        # Truncate if needed
        if self.max_text_length and len(text) > self.max_text_length:
            text = text[:self.max_text_length]

        return text

    def _extract_text(self, element: Optional[etree._Element]) -> str:
        """
        Extract all text from an XML element

        Args:
            element: XML element

        Returns:
            Extracted and cleaned text
        """
        if element is None:
            return ""

        # Get all text content
        if self.remove_formatting:
            # Remove formatting tags, keep only text
            text = ''.join(element.itertext())
        else:
            # Use tostring to preserve some structure
            text = etree.tostring(element, encoding='unicode', method='text')

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def parse_batch(self, xml_files: List[Path], max_workers: int = 4) -> List[ParsedDocument]:
        """
        Parse multiple XML files in parallel

        Args:
            xml_files: List of XML file paths
            max_workers: Number of parallel workers

        Returns:
            List of parsed documents (excluding failed parses)
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        documents = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all parsing tasks
            future_to_file = {
                executor.submit(self.parse_file, xml_file): xml_file
                for xml_file in xml_files
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                xml_file = future_to_file[future]
                try:
                    doc = future.result()
                    if doc:
                        documents.append(doc)
                    else:
                        logger.warning(f"Failed to parse: {xml_file}")
                except Exception as e:
                    logger.error(f"Exception parsing {xml_file}: {e}")

        logger.info(f"Successfully parsed {len(documents)}/{len(xml_files)} documents")
        return documents


def main():
    """Example usage"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Parse FORMEX XML files')
    parser.add_argument('--input', type=str, required=True, help='Input XML file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory for parsed JSON')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize parser
    formex_parser = FORMEXParser()

    # Get list of XML files
    if input_path.is_file():
        xml_files = [input_path]
    else:
        xml_files = list(input_path.rglob('*.xml'))

    logger.info(f"Found {len(xml_files)} XML files to parse")

    # Parse documents
    documents = formex_parser.parse_batch(xml_files, max_workers=args.workers)

    # Save parsed documents
    for doc in documents:
        if doc.metadata.celex:
            filename = f"{doc.metadata.celex}.json"
        else:
            filename = f"document_{hash(doc.full_text)}.json"

        output_file = output_path / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(documents)} parsed documents to {output_path}")


if __name__ == '__main__':
    main()
