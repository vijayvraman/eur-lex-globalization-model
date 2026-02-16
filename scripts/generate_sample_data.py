"""
Generate Sample FORMEX XML Data for Testing

Creates realistic sample EUR-Lex FORMEX XML files for testing the pipeline
without requiring the full 25GB dataset.
"""

import logging
from pathlib import Path
from lxml import etree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_formex_xml(celex: str, language: str, doc_type: str,
                             title: str, num_articles: int = 5) -> str:
    """
    Create a sample FORMEX XML document

    Args:
        celex: CELEX number
        language: Language code (en, fr, de, es, pt)
        doc_type: Document type (regulation, directive, etc.)
        title: Document title
        num_articles: Number of articles to generate

    Returns:
        XML string
    """
    # Create root element
    root = etree.Element('DOCUMENT', LANG=language)

    # Add metadata
    metadata = etree.SubElement(root, 'METADATA')
    celex_elem = etree.SubElement(metadata, 'IDENTIFIER.CELEX')
    celex_elem.text = celex

    doc_type_elem = etree.SubElement(metadata, 'TYPE')
    doc_type_elem.text = doc_type

    date_elem = etree.SubElement(metadata, 'DATE.DOCUMENT')
    date_elem.text = '2024-01-15'

    # Add title
    title_elem = etree.SubElement(root, 'TITLE')
    title_elem.text = title

    # Add body
    body = etree.SubElement(root, 'BODY')

    # Add articles
    for i in range(1, num_articles + 1):
        article = etree.SubElement(body, 'ARTICLE', ID=str(i))

        # Article title
        art_title = etree.SubElement(article, 'TI.ART')
        art_title.text = f"Article {i}"

        # Article paragraphs
        if i == 1:
            # Scope article
            para1 = etree.SubElement(article, 'PARAG')
            para1.text = (
                f"This {doc_type} shall apply to all entities operating within "
                f"the European Union. For the purposes of this {doc_type}, "
                f"'entity' means any legal person or organization."
            )
        elif i == 2:
            # Definitions
            para2 = etree.SubElement(article, 'PARAG')
            para2.text = (
                f"For the purposes of this {doc_type}, the following definitions apply: "
                f"'processing' means any operation performed on data; "
                f"'controller' means the entity that determines the purposes."
            )
        else:
            # Requirements
            para_req = etree.SubElement(article, 'PARAG')
            para_req.text = (
                f"Member States shall ensure compliance with the requirements set out "
                f"in Article {i-1}. Entities must maintain adequate documentation "
                f"and provide regular reports to the competent authority."
            )

    return etree.tostring(root, pretty_print=True, encoding='unicode')


def generate_sample_dataset(output_dir: Path, num_docs_per_lang: int = 5):
    """
    Generate sample dataset in multiple languages

    Args:
        output_dir: Output directory for sample XML files
        num_docs_per_lang: Number of documents per language
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = {
        'en': 'English',
        'fr': 'French',
        'de': 'German',
        'es': 'Spanish',
        'pt': 'Portuguese'
    }

    doc_types = ['regulation', 'directive', 'decision']

    doc_templates = {
        'en': {
            'regulation': 'REGULATION (EU) No {num}/2024 on Data Protection',
            'directive': 'DIRECTIVE (EU) {num}/2024 on Consumer Rights',
            'decision': 'DECISION (EU) {num}/2024 on Market Authorization'
        },
        'fr': {
            'regulation': 'RÈGLEMENT (UE) n° {num}/2024 relatif à la protection des données',
            'directive': 'DIRECTIVE (UE) {num}/2024 relative aux droits des consommateurs',
            'decision': 'DÉCISION (UE) {num}/2024 concernant l\'autorisation de mise sur le marché'
        },
        'de': {
            'regulation': 'VERORDNUNG (EU) Nr. {num}/2024 über Datenschutz',
            'directive': 'RICHTLINIE (EU) {num}/2024 über Verbraucherrechte',
            'decision': 'BESCHLUSS (EU) {num}/2024 über Marktzulassung'
        },
        'es': {
            'regulation': 'REGLAMENTO (UE) n.º {num}/2024 sobre protección de datos',
            'directive': 'DIRECTIVA (UE) {num}/2024 sobre derechos de los consumidores',
            'decision': 'DECISIÓN (UE) {num}/2024 sobre autorización de mercado'
        },
        'pt': {
            'regulation': 'REGULAMENTO (UE) n.º {num}/2024 relativo à proteção de dados',
            'directive': 'DIRETIVA (UE) {num}/2024 relativa aos direitos dos consumidores',
            'decision': 'DECISÃO (UE) {num}/2024 relativa à autorização de mercado'
        }
    }

    total_docs = 0

    for lang_code, lang_name in languages.items():
        lang_dir = output_dir / lang_code
        lang_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {num_docs_per_lang} documents for {lang_name}")

        for doc_idx in range(num_docs_per_lang):
            doc_type = doc_types[doc_idx % len(doc_types)]
            doc_num = 1000 + doc_idx

            # Generate CELEX number
            if doc_type == 'regulation':
                celex = f"32024R{doc_num:04d}"
            elif doc_type == 'directive':
                celex = f"32024L{doc_num:04d}"
            else:
                celex = f"32024D{doc_num:04d}"

            # Get title template
            title = doc_templates[lang_code][doc_type].format(num=doc_num)

            # Create XML
            xml_content = create_sample_formex_xml(
                celex=celex,
                language=lang_code,
                doc_type=doc_type,
                title=title,
                num_articles=5
            )

            # Save to file
            filename = f"{celex}_{lang_code}.xml"
            filepath = lang_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(xml_content)

            total_docs += 1

        logger.info(f"  Created {num_docs_per_lang} documents in {lang_dir}")

    logger.info(f"\nGenerated {total_docs} sample documents in {output_dir}")
    logger.info(f"Languages: {', '.join(languages.keys())}")
    logger.info(f"Documents per language: {num_docs_per_lang}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate sample FORMEX XML data')
    parser.add_argument('--output_dir', type=str, default='data/raw_sample',
                       help='Output directory for sample data')
    parser.add_argument('--num_docs', type=int, default=5,
                       help='Number of documents per language')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    logger.info("Generating sample FORMEX XML dataset...")
    generate_sample_dataset(output_dir, num_docs_per_lang=args.num_docs)
    logger.info("Sample data generation complete!")


if __name__ == '__main__':
    main()
