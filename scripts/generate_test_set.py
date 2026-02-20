"""
Generate Test Set for EUR-Lex Model Comparison

Creates 100 Q&A pairs (20 per language) covering common EUR-Lex legal patterns:
- Definition questions
- Compliance questions
- Requirement questions
- Scope questions
- Citation questions

Output: JSONL file with questions, ground truth answers, and metadata
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def create_english_questions() -> List[Dict]:
    """Create 20 English test questions"""
    questions = [
        # Definition questions (4)
        {
            "question": "What is 'personal data' according to GDPR?",
            "answer": "According to Article 4(1), GDPR (CELEX: 32016R0679), 'personal data' means any information relating to an identified or identifiable natural person ('data subject'); an identifiable natural person is one who can be identified, directly or indirectly.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "What does 'processing' mean under GDPR?",
            "answer": "According to Article 4(2), GDPR (CELEX: 32016R0679), 'processing' means any operation or set of operations performed on personal data, such as collection, recording, organization, structuring, storage, adaptation, retrieval, consultation, use, disclosure, or erasure.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "What is a 'data controller' according to GDPR?",
            "answer": "According to Article 4(7), GDPR (CELEX: 32016R0679), 'controller' means the natural or legal person, public authority, agency or other body which, alone or jointly with others, determines the purposes and means of the processing of personal data.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "What constitutes 'consent' under GDPR?",
            "answer": "According to Article 4(11), GDPR (CELEX: 32016R0679), 'consent' means any freely given, specific, informed and unambiguous indication of the data subject's wishes by which they, by a statement or by a clear affirmative action, signify agreement to the processing of personal data relating to them.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        # Compliance questions (4)
        {
            "question": "What are the key principles for processing personal data under Article 5 of GDPR?",
            "answer": "According to Article 5, GDPR (CELEX: 32016R0679), personal data shall be processed lawfully, fairly and transparently; collected for specified, explicit and legitimate purposes; adequate, relevant and limited to what is necessary; accurate and kept up to date; kept in a form which permits identification for no longer than necessary; and processed in a manner that ensures appropriate security.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "5", "type": "compliance"}
        },
        {
            "question": "What obligations does Article 33 of GDPR impose on controllers regarding data breaches?",
            "answer": "According to Article 33, GDPR (CELEX: 32016R0679), controllers shall notify the supervisory authority of a personal data breach without undue delay and, where feasible, not later than 72 hours after having become aware of it, unless the breach is unlikely to result in a risk to rights and freedoms of natural persons.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "33", "type": "compliance"}
        },
        {
            "question": "What are the controller's obligations under Article 24 of GDPR?",
            "answer": "According to Article 24, GDPR (CELEX: 32016R0679), the controller shall implement appropriate technical and organizational measures to ensure and demonstrate that processing is performed in accordance with the Regulation, taking into account the nature, scope, context and purposes of processing as well as the risks to rights and freedoms of natural persons.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "24", "type": "compliance"}
        },
        {
            "question": "What must organizations do under Article 30 of GDPR?",
            "answer": "According to Article 30, GDPR (CELEX: 32016R0679), each controller and processor shall maintain a record of processing activities under their responsibility, including the purposes of processing, categories of data subjects and personal data, categories of recipients, transfers to third countries, and technical and organizational security measures.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "30", "type": "compliance"}
        },
        # Requirement questions (4)
        {
            "question": "What is required under Article 6 of GDPR for lawful processing?",
            "answer": "According to Article 6, GDPR (CELEX: 32016R0679), processing is lawful only if at least one of the following applies: consent, performance of a contract, compliance with legal obligation, protection of vital interests, performance of a task carried out in the public interest, or legitimate interests pursued by the controller or third party.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "6", "type": "requirement"}
        },
        {
            "question": "What conditions must be met for processing special categories of data under Article 9?",
            "answer": "According to Article 9, GDPR (CELEX: 32016R0679), processing of special categories of personal data is prohibited unless explicit consent is given, processing is necessary for employment or social security law, vital interests, legitimate activities of foundations or associations, data manifestly made public by the data subject, legal claims, substantial public interest, health or social care, public health, or archiving and research purposes.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "9", "type": "requirement"}
        },
        {
            "question": "What is required for transfers of personal data to third countries under Article 45?",
            "answer": "According to Article 45, GDPR (CELEX: 32016R0679), a transfer of personal data to a third country may take place where the European Commission has decided that the third country ensures an adequate level of protection. The adequacy decision shall be based on the rule of law, human rights, data protection rules, and effective enforcement mechanisms.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "45", "type": "requirement"}
        },
        {
            "question": "What must be included in a data protection impact assessment under Article 35?",
            "answer": "According to Article 35, GDPR (CELEX: 32016R0679), a data protection impact assessment shall contain a systematic description of the envisaged processing operations and purposes, an assessment of the necessity and proportionality of processing, an assessment of risks to rights and freedoms of data subjects, and the measures envisaged to address the risks.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "35", "type": "requirement"}
        },
        # Scope questions (4)
        {
            "question": "What is the territorial scope of the GDPR according to Article 3?",
            "answer": "According to Article 3, GDPR (CELEX: 32016R0679), the Regulation applies to the processing of personal data in the context of activities of an establishment of a controller or processor in the EU, regardless of whether processing takes place in the EU. It also applies to processing of personal data of data subjects in the EU by controllers or processors not established in the EU where activities relate to offering goods or services or monitoring behavior within the EU.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "3", "type": "scope"}
        },
        {
            "question": "What is the material scope of GDPR under Article 2?",
            "answer": "According to Article 2, GDPR (CELEX: 32016R0679), the Regulation applies to the processing of personal data wholly or partly by automated means and to other than automated processing of personal data which form part of a filing system or are intended to form part of a filing system. It does not apply to processing in the course of activities outside the scope of EU law, by member states within the scope of common foreign and security policy, by natural persons for purely personal or household activities, or by competent authorities for law enforcement purposes.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "2", "type": "scope"}
        },
        {
            "question": "To whom does the right to be forgotten under Article 17 apply?",
            "answer": "According to Article 17, GDPR (CELEX: 32016R0679), the data subject shall have the right to obtain erasure of personal data concerning them without undue delay where: data is no longer necessary, consent is withdrawn, the data subject objects to processing, data has been unlawfully processed, erasure is required for compliance with legal obligation, or data was collected in relation to information society services offered to children.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "17", "type": "scope"}
        },
        {
            "question": "What rights does Article 15 grant to data subjects?",
            "answer": "According to Article 15, GDPR (CELEX: 32016R0679), the data subject shall have the right to obtain from the controller confirmation as to whether personal data concerning them is being processed, and access to the personal data and information including purposes of processing, categories of personal data, recipients, retention period, rights to rectification or erasure, right to lodge a complaint, source of data, and existence of automated decision-making.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "15", "type": "scope"}
        },
        # Citation questions (4)
        {
            "question": "What does CELEX 32016R0679 refer to?",
            "answer": "CELEX: 32016R0679 refers to Regulation (EU) 2016/679 of the European Parliament and of the Council on the protection of natural persons with regard to the processing of personal data and on the free movement of such data (General Data Protection Regulation - GDPR), which was adopted on 27 April 2016 and applies from 25 May 2018.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "", "type": "citation"}
        },
        {
            "question": "What penalties can be imposed under Article 83 of GDPR?",
            "answer": "According to Article 83, GDPR (CELEX: 32016R0679), administrative fines up to 10,000,000 EUR or 2% of total worldwide annual turnover may be imposed for certain infringements, and fines up to 20,000,000 EUR or 4% of total worldwide annual turnover for more serious violations, such as breach of basic principles for processing, conditions for consent, or data subjects' rights.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "83", "type": "citation"}
        },
        {
            "question": "What does Article 13 of GDPR require controllers to provide to data subjects?",
            "answer": "According to Article 13, GDPR (CELEX: 32016R0679), where personal data is collected from the data subject, the controller shall provide: identity and contact details of controller, purposes and legal basis of processing, recipients of data, retention period, existence of rights, right to withdraw consent, right to lodge complaint, whether provision is statutory or contractual requirement, and existence of automated decision-making.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "13", "type": "citation"}
        },
        {
            "question": "What is the role of the Data Protection Officer under Article 39 of GDPR?",
            "answer": "According to Article 39, GDPR (CELEX: 32016R0679), the Data Protection Officer shall: inform and advise the controller, processor and employees about data protection obligations; monitor compliance with the Regulation; provide advice regarding data protection impact assessments; cooperate with the supervisory authority; and act as the contact point for the supervisory authority on issues relating to processing.",
            "language": "en",
            "metadata": {"celex": "32016R0679", "article": "39", "type": "citation"}
        }
    ]
    return questions


def create_french_questions() -> List[Dict]:
    """Create 20 French test questions"""
    questions = [
        {
            "question": "Qu'est-ce que les 'données personnelles' selon le RGPD?",
            "answer": "Selon l'article 4(1), RGPD (CELEX: 32016R0679), les 'données à caractère personnel' désignent toute information se rapportant à une personne physique identifiée ou identifiable ('personne concernée'); est réputée être une personne physique identifiable une personne qui peut être identifiée, directement ou indirectement.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "Que signifie 'traitement' dans le cadre du RGPD?",
            "answer": "Selon l'article 4(2), RGPD (CELEX: 32016R0679), 'traitement' désigne toute opération ou ensemble d'opérations effectuées sur des données à caractère personnel, telles que la collecte, l'enregistrement, l'organisation, la structuration, la conservation, l'adaptation, la consultation, l'utilisation, la communication ou l'effacement.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "Qu'est-ce qu'un 'responsable du traitement' selon le RGPD?",
            "answer": "Selon l'article 4(7), RGPD (CELEX: 32016R0679), le 'responsable du traitement' est la personne physique ou morale, l'autorité publique, le service ou un autre organisme qui, seul ou conjointement avec d'autres, détermine les finalités et les moyens du traitement de données à caractère personnel.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "Qu'est-ce que le 'consentement' selon le RGPD?",
            "answer": "Selon l'article 4(11), RGPD (CELEX: 32016R0679), le 'consentement' désigne toute manifestation de volonté, libre, spécifique, éclairée et univoque par laquelle la personne concernée accepte, par une déclaration ou par un acte positif clair, que des données à caractère personnel la concernant fassent l'objet d'un traitement.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "Quels sont les principes clés du traitement des données selon l'article 5 du RGPD?",
            "answer": "Selon l'article 5, RGPD (CELEX: 32016R0679), les données personnelles doivent être traitées de manière licite, loyale et transparente; collectées pour des finalités déterminées, explicites et légitimes; adéquates, pertinentes et limitées; exactes et tenues à jour; conservées sous une forme permettant l'identification pendant une durée n'excédant pas celle nécessaire; et traitées de façon à garantir une sécurité appropriée.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "5", "type": "compliance"}
        },
        {
            "question": "Quelles obligations l'article 33 du RGPD impose-t-il aux responsables du traitement en cas de violation de données?",
            "answer": "Selon l'article 33, RGPD (CELEX: 32016R0679), le responsable du traitement notifie à l'autorité de contrôle toute violation de données à caractère personnel dans les meilleurs délais et, si possible, 72 heures au plus tard après en avoir pris connaissance, à moins que la violation ne soit pas susceptible d'engendrer un risque pour les droits et libertés des personnes physiques.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "33", "type": "compliance"}
        },
        {
            "question": "Quelles sont les obligations du responsable selon l'article 24 du RGPD?",
            "answer": "Selon l'article 24, RGPD (CELEX: 32016R0679), le responsable du traitement met en œuvre des mesures techniques et organisationnelles appropriées pour s'assurer et être en mesure de démontrer que le traitement est effectué conformément au Règlement, compte tenu de la nature, de la portée, du contexte et des finalités du traitement ainsi que des risques pour les droits et libertés des personnes.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "24", "type": "compliance"}
        },
        {
            "question": "Que doivent faire les organisations selon l'article 30 du RGPD?",
            "answer": "Selon l'article 30, RGPD (CELEX: 32016R0679), chaque responsable du traitement et sous-traitant tient un registre des activités de traitement effectuées sous leur responsabilité, incluant les finalités du traitement, les catégories de personnes concernées et de données, les catégories de destinataires, les transferts vers des pays tiers, et les mesures de sécurité techniques et organisationnelles.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "30", "type": "compliance"}
        },
        {
            "question": "Qu'est-ce qui est requis selon l'article 6 du RGPD pour un traitement licite?",
            "answer": "Selon l'article 6, RGPD (CELEX: 32016R0679), le traitement n'est licite que si au moins une des conditions suivantes est remplie: consentement, exécution d'un contrat, respect d'une obligation légale, sauvegarde des intérêts vitaux, exécution d'une mission d'intérêt public, ou intérêts légitimes poursuivis par le responsable du traitement ou par un tiers.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "6", "type": "requirement"}
        },
        {
            "question": "Quelles conditions doivent être remplies pour traiter des catégories particulières de données selon l'article 9?",
            "answer": "Selon l'article 9, RGPD (CELEX: 32016R0679), le traitement de catégories particulières de données est interdit sauf si: consentement explicite donné, traitement nécessaire au droit du travail ou de sécurité sociale, intérêts vitaux, activités légitimes de fondations ou associations, données manifestement rendues publiques, constatation d'un droit, motif d'intérêt public important, médecine préventive ou santé publique, ou archivage et recherche.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "9", "type": "requirement"}
        },
        {
            "question": "Qu'est-ce qui est requis pour les transferts de données vers des pays tiers selon l'article 45?",
            "answer": "Selon l'article 45, RGPD (CELEX: 32016R0679), un transfert de données vers un pays tiers peut avoir lieu lorsque la Commission européenne a constaté que le pays tiers assure un niveau de protection adéquat. La décision d'adéquation est fondée sur l'état de droit, les droits de l'homme, les règles de protection des données, et l'effectivité des mécanismes d'application.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "45", "type": "requirement"}
        },
        {
            "question": "Que doit contenir une analyse d'impact relative à la protection des données selon l'article 35?",
            "answer": "Selon l'article 35, RGPD (CELEX: 32016R0679), une analyse d'impact doit contenir une description systématique des opérations de traitement envisagées et des finalités, une évaluation de la nécessité et de la proportionnalité du traitement, une évaluation des risques pour les droits et libertés des personnes, et les mesures envisagées pour faire face aux risques.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "35", "type": "requirement"}
        },
        {
            "question": "Quel est le champ d'application territorial du RGPD selon l'article 3?",
            "answer": "Selon l'article 3, RGPD (CELEX: 32016R0679), le Règlement s'applique au traitement de données effectué dans le cadre des activités d'un établissement d'un responsable ou sous-traitant sur le territoire de l'UE. Il s'applique aussi au traitement de données de personnes dans l'UE par des responsables ou sous-traitants non établis dans l'UE, lorsque les activités sont liées à l'offre de biens ou services ou au suivi du comportement dans l'UE.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "3", "type": "scope"}
        },
        {
            "question": "Quel est le champ d'application matériel du RGPD selon l'article 2?",
            "answer": "Selon l'article 2, RGPD (CELEX: 32016R0679), le Règlement s'applique au traitement de données automatisé en tout ou en partie, ainsi qu'au traitement non automatisé de données faisant partie d'un fichier ou destinées à en faire partie. Il ne s'applique pas au traitement effectué dans le cadre d'activités hors du champ du droit de l'UE, par les États membres dans le cadre de la politique étrangère et de sécurité commune, par des personnes physiques dans le cadre d'activités strictement personnelles, ou par les autorités compétentes à des fins répressives.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "2", "type": "scope"}
        },
        {
            "question": "À qui s'applique le droit à l'effacement selon l'article 17?",
            "answer": "Selon l'article 17, RGPD (CELEX: 32016R0679), la personne concernée a le droit d'obtenir l'effacement de données la concernant sans retard injustifié lorsque: les données ne sont plus nécessaires, le consentement est retiré, la personne s'oppose au traitement, les données ont été traitées illicitement, l'effacement est requis pour respecter une obligation légale, ou les données ont été collectées dans le cadre de services de la société de l'information offerts à des enfants.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "17", "type": "scope"}
        },
        {
            "question": "Quels droits l'article 15 accorde-t-il aux personnes concernées?",
            "answer": "Selon l'article 15, RGPD (CELEX: 32016R0679), la personne concernée a le droit d'obtenir du responsable la confirmation que des données la concernant sont ou ne sont pas traitées, et l'accès aux données ainsi qu'aux informations sur les finalités du traitement, les catégories de données, les destinataires, la durée de conservation, les droits à rectification ou effacement, le droit de déposer une plainte, la source des données, et l'existence d'une prise de décision automatisée.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "15", "type": "scope"}
        },
        {
            "question": "À quoi fait référence CELEX 32016R0679?",
            "answer": "CELEX: 32016R0679 fait référence au Règlement (UE) 2016/679 du Parlement européen et du Conseil relatif à la protection des personnes physiques à l'égard du traitement des données à caractère personnel et à la libre circulation de ces données (Règlement général sur la protection des données - RGPD), adopté le 27 avril 2016 et applicable depuis le 25 mai 2018.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "", "type": "citation"}
        },
        {
            "question": "Quelles sanctions peuvent être imposées selon l'article 83 du RGPD?",
            "answer": "Selon l'article 83, RGPD (CELEX: 32016R0679), des amendes administratives allant jusqu'à 10 000 000 EUR ou 2% du chiffre d'affaires annuel mondial total peuvent être imposées pour certaines infractions, et des amendes allant jusqu'à 20 000 000 EUR ou 4% du chiffre d'affaires annuel mondial total pour les violations les plus graves, telles que la violation des principes de base du traitement, des conditions du consentement, ou des droits des personnes concernées.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "83", "type": "citation"}
        },
        {
            "question": "Que l'article 13 du RGPD exige-t-il que les responsables fournissent aux personnes concernées?",
            "answer": "Selon l'article 13, RGPD (CELEX: 32016R0679), lorsque des données sont collectées auprès de la personne concernée, le responsable fournit: l'identité et coordonnées du responsable, les finalités et base juridique du traitement, les destinataires des données, la durée de conservation, l'existence de droits, le droit de retirer le consentement, le droit de déposer une plainte, si la fourniture est obligatoire, et l'existence d'une prise de décision automatisée.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "13", "type": "citation"}
        },
        {
            "question": "Quel est le rôle du délégué à la protection des données selon l'article 39 du RGPD?",
            "answer": "Selon l'article 39, RGPD (CELEX: 32016R0679), le délégué à la protection des données: informe et conseille le responsable, le sous-traitant et les employés sur les obligations de protection des données; contrôle le respect du Règlement; fournit des conseils sur les analyses d'impact; coopère avec l'autorité de contrôle; et fait office de point de contact pour l'autorité de contrôle sur les questions relatives au traitement.",
            "language": "fr",
            "metadata": {"celex": "32016R0679", "article": "39", "type": "citation"}
        }
    ]
    return questions


def create_german_questions() -> List[Dict]:
    """Create 20 German test questions"""
    questions = [
        {
            "question": "Was sind 'personenbezogene Daten' gemäß DSGVO?",
            "answer": "Gemäß Artikel 4(1), DSGVO (CELEX: 32016R0679), sind 'personenbezogene Daten' alle Informationen, die sich auf eine identifizierte oder identifizierbare natürliche Person ('betroffene Person') beziehen; als identifizierbar wird eine natürliche Person angesehen, die direkt oder indirekt identifiziert werden kann.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "Was bedeutet 'Verarbeitung' im Rahmen der DSGVO?",
            "answer": "Gemäß Artikel 4(2), DSGVO (CELEX: 32016R0679), bedeutet 'Verarbeitung' jeden mit oder ohne Hilfe automatisierter Verfahren ausgeführten Vorgang im Zusammenhang mit personenbezogenen Daten, wie das Erheben, das Erfassen, die Organisation, das Ordnen, die Speicherung, die Anpassung, das Auslesen, die Verwendung, die Offenlegung oder das Löschen.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "Was ist ein 'Verantwortlicher' gemäß DSGVO?",
            "answer": "Gemäß Artikel 4(7), DSGVO (CELEX: 32016R0679), ist der 'Verantwortliche' die natürliche oder juristische Person, Behörde, Einrichtung oder andere Stelle, die allein oder gemeinsam mit anderen über die Zwecke und Mittel der Verarbeitung von personenbezogenen Daten entscheidet.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "Was stellt 'Einwilligung' gemäß DSGVO dar?",
            "answer": "Gemäß Artikel 4(11), DSGVO (CELEX: 32016R0679), ist 'Einwilligung' jede freiwillig für den bestimmten Fall, in informierter Weise und unmissverständlich abgegebene Willensbekundung in Form einer Erklärung oder sonstigen eindeutigen bestätigenden Handlung, mit der die betroffene Person zu verstehen gibt, dass sie mit der Verarbeitung der sie betreffenden personenbezogenen Daten einverstanden ist.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "Was sind die Grundsätze für die Verarbeitung personenbezogener Daten gemäß Artikel 5 der DSGVO?",
            "answer": "Gemäß Artikel 5, DSGVO (CELEX: 32016R0679), müssen personenbezogene Daten auf rechtmäßige, faire und transparente Weise verarbeitet werden; für festgelegte, eindeutige und legitime Zwecke erhoben werden; dem Zweck angemessen, erheblich und auf das notwendige Maß beschränkt sein; sachlich richtig und auf dem neuesten Stand sein; in einer Form gespeichert werden, die die Identifizierung nur so lange ermöglicht, wie es erforderlich ist; und in einer Weise verarbeitet werden, die eine angemessene Sicherheit gewährleistet.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "5", "type": "compliance"}
        },
        {
            "question": "Welche Verpflichtungen erlegt Artikel 33 der DSGVO den Verantwortlichen bei Datenschutzverletzungen auf?",
            "answer": "Gemäß Artikel 33, DSGVO (CELEX: 32016R0679), muss der Verantwortliche unverzüglich und möglichst binnen 72 Stunden, nachdem ihm die Verletzung bekannt wurde, diese der Aufsichtsbehörde melden, es sei denn, dass die Verletzung des Schutzes personenbezogener Daten voraussichtlich nicht zu einem Risiko für die Rechte und Freiheiten natürlicher Personen führt.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "33", "type": "compliance"}
        },
        {
            "question": "Welche Verpflichtungen hat der Verantwortliche gemäß Artikel 24 der DSGVO?",
            "answer": "Gemäß Artikel 24, DSGVO (CELEX: 32016R0679), muss der Verantwortliche geeignete technische und organisatorische Maßnahmen umsetzen, um sicherzustellen und nachweisen zu können, dass die Verarbeitung gemäß der Verordnung erfolgt, unter Berücksichtigung der Art, des Umfangs, der Umstände und der Zwecke der Verarbeitung sowie der Risiken für die Rechte und Freiheiten natürlicher Personen.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "24", "type": "compliance"}
        },
        {
            "question": "Was müssen Organisationen gemäß Artikel 30 der DSGVO tun?",
            "answer": "Gemäß Artikel 30, DSGVO (CELEX: 32016R0679), muss jeder Verantwortliche und Auftragsverarbeiter ein Verzeichnis aller Verarbeitungstätigkeiten führen, das die Zwecke der Verarbeitung, die Kategorien betroffener Personen und personenbezogener Daten, die Kategorien von Empfängern, Übermittlungen an Drittländer und technische und organisatorische Sicherheitsmaßnahmen umfasst.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "30", "type": "compliance"}
        },
        {
            "question": "Was ist gemäß Artikel 6 der DSGVO für eine rechtmäßige Verarbeitung erforderlich?",
            "answer": "Gemäß Artikel 6, DSGVO (CELEX: 32016R0679), ist die Verarbeitung nur rechtmäßig, wenn mindestens eine der folgenden Bedingungen erfüllt ist: Einwilligung, Erfüllung eines Vertrags, Erfüllung einer rechtlichen Verpflichtung, Schutz lebenswichtiger Interessen, Wahrnehmung einer Aufgabe im öffentlichen Interesse oder berechtigte Interessen des Verantwortlichen oder eines Dritten.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "6", "type": "requirement"}
        },
        {
            "question": "Welche Bedingungen müssen für die Verarbeitung besonderer Kategorien von Daten gemäß Artikel 9 erfüllt sein?",
            "answer": "Gemäß Artikel 9, DSGVO (CELEX: 32016R0679), ist die Verarbeitung besonderer Kategorien personenbezogener Daten untersagt, es sei denn: ausdrückliche Einwilligung liegt vor, Verarbeitung ist für Arbeits- oder Sozialversicherungsrecht erforderlich, lebenswichtige Interessen, rechtmäßige Tätigkeiten von Stiftungen oder Vereinigungen, Daten wurden von der betroffenen Person offenkundig öffentlich gemacht, Geltendmachung von Rechtsansprüchen, erhebliches öffentliches Interesse, Gesundheitsvorsorge oder öffentliche Gesundheit, oder Archivzwecke und Forschung.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "9", "type": "requirement"}
        },
        {
            "question": "Was ist für Übermittlungen personenbezogener Daten an Drittländer gemäß Artikel 45 erforderlich?",
            "answer": "Gemäß Artikel 45, DSGVO (CELEX: 32016R0679), darf eine Übermittlung personenbezogener Daten an ein Drittland erfolgen, wenn die Europäische Kommission beschlossen hat, dass das Drittland ein angemessenes Schutzniveau gewährleistet. Der Angemessenheitsbeschluss beruht auf der Rechtsstaatlichkeit, den Menschenrechten, den Datenschutzvorschriften und wirksamen Durchsetzungsmechanismen.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "45", "type": "requirement"}
        },
        {
            "question": "Was muss eine Datenschutz-Folgenabschätzung gemäß Artikel 35 enthalten?",
            "answer": "Gemäß Artikel 35, DSGVO (CELEX: 32016R0679), muss eine Datenschutz-Folgenabschätzung eine systematische Beschreibung der geplanten Verarbeitungsvorgänge und der Zwecke, eine Bewertung der Notwendigkeit und Verhältnismäßigkeit der Verarbeitung, eine Bewertung der Risiken für die Rechte und Freiheiten der betroffenen Personen und die zur Bewältigung der Risiken geplanten Abhilfemaßnahmen enthalten.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "35", "type": "requirement"}
        },
        {
            "question": "Was ist der räumliche Anwendungsbereich der DSGVO gemäß Artikel 3?",
            "answer": "Gemäß Artikel 3, DSGVO (CELEX: 32016R0679), gilt die Verordnung für die Verarbeitung personenbezogener Daten im Rahmen der Tätigkeiten einer Niederlassung eines Verantwortlichen oder Auftragsverarbeiters in der EU. Sie gilt auch für die Verarbeitung personenbezogener Daten von betroffenen Personen in der EU durch nicht in der EU niedergelassene Verantwortliche oder Auftragsverarbeiter, wenn die Tätigkeiten mit dem Angebot von Waren oder Dienstleistungen oder der Beobachtung des Verhaltens in der EU zusammenhängen.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "3", "type": "scope"}
        },
        {
            "question": "Was ist der sachliche Anwendungsbereich der DSGVO gemäß Artikel 2?",
            "answer": "Gemäß Artikel 2, DSGVO (CELEX: 32016R0679), gilt die Verordnung für die ganz oder teilweise automatisierte Verarbeitung personenbezogener Daten sowie für die nichtautomatisierte Verarbeitung personenbezogener Daten, die in einem Dateisystem gespeichert sind oder gespeichert werden sollen. Sie gilt nicht für die Verarbeitung im Rahmen von Tätigkeiten außerhalb des Anwendungsbereichs des Unionsrechts, durch die Mitgliedstaaten im Rahmen der gemeinsamen Außen- und Sicherheitspolitik, durch natürliche Personen zur Ausübung ausschließlich persönlicher oder familiärer Tätigkeiten oder durch zuständige Behörden zum Zwecke der Verhütung und Verfolgung von Straftaten.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "2", "type": "scope"}
        },
        {
            "question": "Für wen gilt das Recht auf Vergessenwerden gemäß Artikel 17?",
            "answer": "Gemäß Artikel 17, DSGVO (CELEX: 32016R0679), hat die betroffene Person das Recht, von dem Verantwortlichen unverzüglich die Löschung sie betreffender personenbezogener Daten zu verlangen, wenn: die Daten nicht mehr notwendig sind, die Einwilligung widerrufen wird, die betroffene Person Widerspruch einlegt, die Daten unrechtmäßig verarbeitet wurden, die Löschung zur Erfüllung einer rechtlichen Verpflichtung erforderlich ist oder die Daten in Bezug auf Dienste der Informationsgesellschaft an Kinder erhoben wurden.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "17", "type": "scope"}
        },
        {
            "question": "Welche Rechte gewährt Artikel 15 den betroffenen Personen?",
            "answer": "Gemäß Artikel 15, DSGVO (CELEX: 32016R0679), hat die betroffene Person das Recht, von dem Verantwortlichen eine Bestätigung darüber zu verlangen, ob sie betreffende personenbezogene Daten verarbeitet werden, und Zugang zu den Daten sowie zu Informationen über die Zwecke der Verarbeitung, die Kategorien personenbezogener Daten, die Empfänger, die Speicherdauer, die Rechte auf Berichtigung oder Löschung, das Recht auf Beschwerde, die Herkunft der Daten und das Bestehen einer automatisierten Entscheidungsfindung.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "15", "type": "scope"}
        },
        {
            "question": "Worauf bezieht sich CELEX 32016R0679?",
            "answer": "CELEX: 32016R0679 bezieht sich auf die Verordnung (EU) 2016/679 des Europäischen Parlaments und des Rates zum Schutz natürlicher Personen bei der Verarbeitung personenbezogener Daten und zum freien Datenverkehr (Datenschutz-Grundverordnung - DSGVO), die am 27. April 2016 angenommen wurde und seit dem 25. Mai 2018 gilt.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "", "type": "citation"}
        },
        {
            "question": "Welche Geldbußen können gemäß Artikel 83 der DSGVO verhängt werden?",
            "answer": "Gemäß Artikel 83, DSGVO (CELEX: 32016R0679), können Geldbußen von bis zu 10.000.000 EUR oder 2% des gesamten weltweit erzielten Jahresumsatzes für bestimmte Verstöße verhängt werden, und Geldbußen von bis zu 20.000.000 EUR oder 4% des gesamten weltweit erzielten Jahresumsatzes für schwerwiegendere Verstöße, wie die Verletzung der Grundsätze für die Verarbeitung, der Bedingungen für die Einwilligung oder der Rechte der betroffenen Person.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "83", "type": "citation"}
        },
        {
            "question": "Was verlangt Artikel 13 der DSGVO von Verantwortlichen gegenüber betroffenen Personen?",
            "answer": "Gemäß Artikel 13, DSGVO (CELEX: 32016R0679), muss der Verantwortliche, wenn personenbezogene Daten bei der betroffenen Person erhoben werden, Folgendes bereitstellen: Identität und Kontaktdaten des Verantwortlichen, Zwecke und Rechtsgrundlage der Verarbeitung, Empfänger der Daten, Speicherdauer, Bestehen von Rechten, Recht auf Widerruf der Einwilligung, Beschwerderecht, ob die Bereitstellung gesetzlich oder vertraglich vorgeschrieben ist, und das Bestehen einer automatisierten Entscheidungsfindung.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "13", "type": "citation"}
        },
        {
            "question": "Welche Aufgabe hat der Datenschutzbeauftragte gemäß Artikel 39 der DSGVO?",
            "answer": "Gemäß Artikel 39, DSGVO (CELEX: 32016R0679), hat der Datenschutzbeauftragte folgende Aufgaben: Unterrichtung und Beratung des Verantwortlichen, Auftragsverarbeiters und der Beschäftigten über datenschutzrechtliche Pflichten; Überwachung der Einhaltung der Verordnung; Beratung bei Datenschutz-Folgenabschätzungen; Zusammenarbeit mit der Aufsichtsbehörde; und Funktion als Anlaufstelle für die Aufsichtsbehörde in mit der Verarbeitung zusammenhängenden Fragen.",
            "language": "de",
            "metadata": {"celex": "32016R0679", "article": "39", "type": "citation"}
        }
    ]
    return questions


def create_spanish_questions() -> List[Dict]:
    """Create 20 Spanish test questions"""
    questions = [
        {
            "question": "¿Qué son los 'datos personales' según el RGPD?",
            "answer": "Según el Artículo 4(1), RGPD (CELEX: 32016R0679), 'datos personales' significa toda información sobre una persona física identificada o identificable ('interesado'); se considerará persona física identificable toda persona cuya identidad pueda determinarse, directa o indirectamente.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "¿Qué significa 'tratamiento' en el marco del RGPD?",
            "answer": "Según el Artículo 4(2), RGPD (CELEX: 32016R0679), 'tratamiento' significa cualquier operación o conjunto de operaciones realizadas sobre datos personales, como la recogida, registro, organización, estructuración, conservación, adaptación, consulta, utilización, comunicación o supresión.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "¿Qué es un 'responsable del tratamiento' según el RGPD?",
            "answer": "Según el Artículo 4(7), RGPD (CELEX: 32016R0679), 'responsable del tratamiento' es la persona física o jurídica, autoridad pública, servicio u otro organismo que, solo o junto con otros, determine los fines y medios del tratamiento de datos personales.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "¿Qué constituye 'consentimiento' según el RGPD?",
            "answer": "Según el Artículo 4(11), RGPD (CELEX: 32016R0679), 'consentimiento' significa toda manifestación de voluntad libre, específica, informada e inequívoca por la que el interesado acepta, ya sea mediante una declaración o una clara acción afirmativa, el tratamiento de datos personales que le conciernen.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "¿Cuáles son los principios clave para el tratamiento de datos personales según el Artículo 5 del RGPD?",
            "answer": "Según el Artículo 5, RGPD (CELEX: 32016R0679), los datos personales serán tratados de manera lícita, leal y transparente; recogidos con fines determinados, explícitos y legítimos; adecuados, pertinentes y limitados; exactos y actualizados; mantenidos de forma que permita la identificación durante no más tiempo del necesario; y tratados de manera que garantice una seguridad adecuada.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "5", "type": "compliance"}
        },
        {
            "question": "¿Qué obligaciones impone el Artículo 33 del RGPD a los responsables en caso de violación de datos?",
            "answer": "Según el Artículo 33, RGPD (CELEX: 32016R0679), el responsable del tratamiento notificará a la autoridad de control cualquier violación de la seguridad de los datos personales sin dilación indebida y, de ser posible, a más tardar 72 horas después de tener constancia de ella, a menos que sea improbable que la violación constituya un riesgo para los derechos y libertades de las personas físicas.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "33", "type": "compliance"}
        },
        {
            "question": "¿Cuáles son las obligaciones del responsable según el Artículo 24 del RGPD?",
            "answer": "Según el Artículo 24, RGPD (CELEX: 32016R0679), el responsable del tratamiento aplicará medidas técnicas y organizativas apropiadas para garantizar y poder demostrar que el tratamiento es conforme con el Reglamento, teniendo en cuenta la naturaleza, el alcance, el contexto y los fines del tratamiento, así como los riesgos para los derechos y libertades de las personas.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "24", "type": "compliance"}
        },
        {
            "question": "¿Qué deben hacer las organizaciones según el Artículo 30 del RGPD?",
            "answer": "Según el Artículo 30, RGPD (CELEX: 32016R0679), cada responsable y encargado del tratamiento llevará un registro de las actividades de tratamiento efectuadas bajo su responsabilidad, que incluya los fines del tratamiento, las categorías de interesados y de datos personales, las categorías de destinatarios, las transferencias a terceros países, y las medidas técnicas y organizativas de seguridad.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "30", "type": "compliance"}
        },
        {
            "question": "¿Qué se requiere según el Artículo 6 del RGPD para un tratamiento lícito?",
            "answer": "Según el Artículo 6, RGPD (CELEX: 32016R0679), el tratamiento solo será lícito si se cumple al menos una de las siguientes condiciones: consentimiento, ejecución de un contrato, cumplimiento de una obligación legal, protección de intereses vitales, cumplimiento de una misión realizada en interés público, o intereses legítimos perseguidos por el responsable del tratamiento o por un tercero.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "6", "type": "requirement"}
        },
        {
            "question": "¿Qué condiciones deben cumplirse para tratar categorías especiales de datos según el Artículo 9?",
            "answer": "Según el Artículo 9, RGPD (CELEX: 32016R0679), el tratamiento de categorías especiales de datos está prohibido salvo que: se dé consentimiento explícito, el tratamiento sea necesario para derecho laboral o de seguridad social, intereses vitales, actividades legítimas de fundaciones o asociaciones, datos hechos públicos manifiestamente por el interesado, reclamaciones, interés público importante, medicina preventiva o salud pública, o fines de archivo e investigación.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "9", "type": "requirement"}
        },
        {
            "question": "¿Qué se requiere para transferencias de datos personales a terceros países según el Artículo 45?",
            "answer": "Según el Artículo 45, RGPD (CELEX: 32016R0679), una transferencia de datos personales a un tercer país podrá tener lugar cuando la Comisión Europea haya decidido que el tercer país garantiza un nivel de protección adecuado. La decisión de adecuación se basará en el Estado de Derecho, los derechos humanos, las normas de protección de datos y la efectividad de los mecanismos de aplicación.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "45", "type": "requirement"}
        },
        {
            "question": "¿Qué debe contener una evaluación de impacto relativa a la protección de datos según el Artículo 35?",
            "answer": "Según el Artículo 35, RGPD (CELEX: 32016R0679), una evaluación de impacto deberá contener una descripción sistemática de las operaciones de tratamiento previstas y de los fines del tratamiento, una evaluación de la necesidad y proporcionalidad del tratamiento, una evaluación de los riesgos para los derechos y libertades de los interesados, y las medidas previstas para afrontar los riesgos.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "35", "type": "requirement"}
        },
        {
            "question": "¿Cuál es el ámbito de aplicación territorial del RGPD según el Artículo 3?",
            "answer": "Según el Artículo 3, RGPD (CELEX: 32016R0679), el Reglamento se aplica al tratamiento de datos personales en el contexto de las actividades de un establecimiento del responsable o del encargado en la UE. También se aplica al tratamiento de datos personales de interesados en la UE por parte de responsables o encargados no establecidos en la UE, cuando las actividades estén relacionadas con la oferta de bienes o servicios o el control del comportamiento en la UE.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "3", "type": "scope"}
        },
        {
            "question": "¿Cuál es el ámbito de aplicación material del RGPD según el Artículo 2?",
            "answer": "Según el Artículo 2, RGPD (CELEX: 32016R0679), el Reglamento se aplica al tratamiento total o parcialmente automatizado de datos personales, así como al tratamiento no automatizado de datos que formen parte de un fichero o estén destinados a formar parte de un fichero. No se aplica al tratamiento efectuado en el curso de actividades no comprendidas en el ámbito del Derecho de la UE, por los Estados miembros en el ámbito de la política exterior y de seguridad común, por personas físicas en el ejercicio de actividades exclusivamente personales o domésticas, o por autoridades competentes con fines de prevención o enjuiciamiento de infracciones penales.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "2", "type": "scope"}
        },
        {
            "question": "¿A quién se aplica el derecho al olvido según el Artículo 17?",
            "answer": "Según el Artículo 17, RGPD (CELEX: 32016R0679), el interesado tendrá derecho a obtener sin dilación indebida del responsable la supresión de datos personales que le conciernan cuando: los datos ya no sean necesarios, se retire el consentimiento, el interesado se oponga al tratamiento, los datos se hayan tratado ilícitamente, sea necesario suprimir los datos para cumplir una obligación legal, o los datos se hayan obtenido en relación con servicios de la sociedad de la información ofrecidos a niños.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "17", "type": "scope"}
        },
        {
            "question": "¿Qué derechos otorga el Artículo 15 a los interesados?",
            "answer": "Según el Artículo 15, RGPD (CELEX: 32016R0679), el interesado tendrá derecho a obtener del responsable la confirmación de si se están tratando datos personales que le conciernen y, en tal caso, acceso a los datos y a información sobre los fines del tratamiento, las categorías de datos, los destinatarios, el plazo de conservación, los derechos de rectificación o supresión, el derecho a presentar una reclamación, el origen de los datos, y la existencia de decisiones automatizadas.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "15", "type": "scope"}
        },
        {
            "question": "¿A qué hace referencia CELEX 32016R0679?",
            "answer": "CELEX: 32016R0679 hace referencia al Reglamento (UE) 2016/679 del Parlamento Europeo y del Consejo relativo a la protección de las personas físicas en lo que respecta al tratamiento de datos personales y a la libre circulación de estos datos (Reglamento General de Protección de Datos - RGPD), adoptado el 27 de abril de 2016 y aplicable desde el 25 de mayo de 2018.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "", "type": "citation"}
        },
        {
            "question": "¿Qué sanciones pueden imponerse según el Artículo 83 del RGPD?",
            "answer": "Según el Artículo 83, RGPD (CELEX: 32016R0679), las multas administrativas pueden ser de hasta 10.000.000 EUR o el 2% del volumen de negocio total anual global para ciertas infracciones, y multas de hasta 20.000.000 EUR o el 4% del volumen de negocio total anual global para las infracciones más graves, como la violación de los principios básicos del tratamiento, las condiciones para el consentimiento o los derechos de los interesados.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "83", "type": "citation"}
        },
        {
            "question": "¿Qué requiere el Artículo 13 del RGPD que los responsables proporcionen a los interesados?",
            "answer": "Según el Artículo 13, RGPD (CELEX: 32016R0679), cuando se obtengan datos personales del interesado, el responsable deberá facilitar: la identidad y datos de contacto del responsable, los fines y base jurídica del tratamiento, los destinatarios de los datos, el plazo de conservación, la existencia de derechos, el derecho a retirar el consentimiento, el derecho a presentar una reclamación, si la comunicación de datos es obligatoria, y la existencia de decisiones automatizadas.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "13", "type": "citation"}
        },
        {
            "question": "¿Cuál es la función del delegado de protección de datos según el Artículo 39 del RGPD?",
            "answer": "Según el Artículo 39, RGPD (CELEX: 32016R0679), el delegado de protección de datos: informará y asesorará al responsable, encargado y empleados sobre las obligaciones de protección de datos; supervisará el cumplimiento del Reglamento; ofrecerá asesoramiento sobre las evaluaciones de impacto; cooperará con la autoridad de control; y actuará como punto de contacto de la autoridad de control en cuestiones relativas al tratamiento.",
            "language": "es",
            "metadata": {"celex": "32016R0679", "article": "39", "type": "citation"}
        }
    ]
    return questions


def create_portuguese_questions() -> List[Dict]:
    """Create 20 Portuguese test questions"""
    questions = [
        {
            "question": "O que são 'dados pessoais' de acordo com o RGPD?",
            "answer": "De acordo com o Artigo 4(1), RGPD (CELEX: 32016R0679), 'dados pessoais' significa informação relativa a uma pessoa singular identificada ou identificável ('titular dos dados'); é considerada identificável uma pessoa singular que possa ser identificada, direta ou indiretamente.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "O que significa 'tratamento' no âmbito do RGPD?",
            "answer": "De acordo com o Artigo 4(2), RGPD (CELEX: 32016R0679), 'tratamento' significa uma operação ou um conjunto de operações efetuadas sobre dados pessoais, como a recolha, o registo, a organização, a estruturação, a conservação, a adaptação, a consulta, a utilização, a divulgação ou a eliminação.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "O que é um 'responsável pelo tratamento' de acordo com o RGPD?",
            "answer": "De acordo com o Artigo 4(7), RGPD (CELEX: 32016R0679), 'responsável pelo tratamento' é a pessoa singular ou coletiva, a autoridade pública, a agência ou outro organismo que, individualmente ou em conjunto com outras, determina as finalidades e os meios de tratamento de dados pessoais.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "O que constitui 'consentimento' de acordo com o RGPD?",
            "answer": "De acordo com o Artigo 4(11), RGPD (CELEX: 32016R0679), 'consentimento' significa uma manifestação de vontade, livre, específica, informada e inequívoca, pela qual o titular dos dados aceita, mediante declaração ou ato positivo inequívoco, que os dados pessoais que lhe dizem respeito sejam objeto de tratamento.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "4", "type": "definition"}
        },
        {
            "question": "Quais são os princípios-chave para o tratamento de dados pessoais de acordo com o Artigo 5 do RGPD?",
            "answer": "De acordo com o Artigo 5, RGPD (CELEX: 32016R0679), os dados pessoais são tratados de forma lícita, leal e transparente; recolhidos para finalidades determinadas, explícitas e legítimas; adequados, pertinentes e limitados; exatos e atualizados; conservados de uma forma que permita a identificação apenas durante o período necessário; e tratados de forma a garantir uma segurança adequada.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "5", "type": "compliance"}
        },
        {
            "question": "Que obrigações o Artigo 33 do RGPD impõe aos responsáveis em caso de violação de dados?",
            "answer": "De acordo com o Artigo 33, RGPD (CELEX: 32016R0679), o responsável pelo tratamento notifica a autoridade de controlo de uma violação de dados pessoais no prazo de 72 horas após ter tido conhecimento da mesma, a menos que seja improvável que a violação constitua um risco para os direitos e liberdades das pessoas singulares.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "33", "type": "compliance"}
        },
        {
            "question": "Quais são as obrigações do responsável de acordo com o Artigo 24 do RGPD?",
            "answer": "De acordo com o Artigo 24, RGPD (CELEX: 32016R0679), o responsável pelo tratamento aplica medidas técnicas e organizativas adequadas para assegurar e poder demonstrar que o tratamento é realizado em conformidade com o Regulamento, tendo em conta a natureza, o âmbito, o contexto e as finalidades do tratamento, bem como os riscos para os direitos e liberdades das pessoas.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "24", "type": "compliance"}
        },
        {
            "question": "O que as organizações devem fazer de acordo com o Artigo 30 do RGPD?",
            "answer": "De acordo com o Artigo 30, RGPD (CELEX: 32016R0679), cada responsável e subcontratante mantém um registo das atividades de tratamento sob a sua responsabilidade, incluindo as finalidades do tratamento, as categorias de titulares de dados e de dados pessoais, as categorias de destinatários, as transferências para países terceiros, e as medidas de segurança técnicas e organizacionais.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "30", "type": "compliance"}
        },
        {
            "question": "O que é necessário de acordo com o Artigo 6 do RGPD para um tratamento lícito?",
            "answer": "De acordo com o Artigo 6, RGPD (CELEX: 32016R0679), o tratamento só é lícito se pelo menos uma das seguintes condições se aplicar: consentimento, execução de um contrato, cumprimento de uma obrigação jurídica, defesa de interesses vitais, execução de uma missão de interesse público, ou interesses legítimos prosseguidos pelo responsável pelo tratamento ou por um terceiro.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "6", "type": "requirement"}
        },
        {
            "question": "Que condições devem ser cumpridas para tratar categorias especiais de dados de acordo com o Artigo 9?",
            "answer": "De acordo com o Artigo 9, RGPD (CELEX: 32016R0679), o tratamento de categorias especiais de dados é proibido, salvo se: for dado consentimento explícito, o tratamento for necessário para direito do trabalho ou segurança social, interesses vitais, atividades legítimas de fundações ou associações, dados manifestamente tornados públicos pelo titular, tutela de direitos, interesse público importante, medicina preventiva ou saúde pública, ou fins de arquivo e investigação.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "9", "type": "requirement"}
        },
        {
            "question": "O que é necessário para transferências de dados pessoais para países terceiros de acordo com o Artigo 45?",
            "answer": "De acordo com o Artigo 45, RGPD (CELEX: 32016R0679), uma transferência de dados pessoais para um país terceiro pode ter lugar quando a Comissão Europeia decidir que o país terceiro assegura um nível de proteção adequado. A decisão de adequação baseia-se no Estado de direito, nos direitos humanos, nas regras de proteção de dados e na eficácia dos mecanismos de aplicação.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "45", "type": "requirement"}
        },
        {
            "question": "O que deve conter uma avaliação de impacto sobre a proteção de dados de acordo com o Artigo 35?",
            "answer": "De acordo com o Artigo 35, RGPD (CELEX: 32016R0679), uma avaliação de impacto deve conter uma descrição sistemática das operações de tratamento previstas e das finalidades do tratamento, uma avaliação da necessidade e proporcionalidade do tratamento, uma avaliação dos riscos para os direitos e liberdades dos titulares dos dados, e as medidas previstas para fazer face aos riscos.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "35", "type": "requirement"}
        },
        {
            "question": "Qual é o âmbito de aplicação territorial do RGPD de acordo com o Artigo 3?",
            "answer": "De acordo com o Artigo 3, RGPD (CELEX: 32016R0679), o Regulamento aplica-se ao tratamento de dados pessoais no contexto das atividades de um estabelecimento de um responsável ou subcontratante na UE. Aplica-se também ao tratamento de dados pessoais de titulares na UE por responsáveis ou subcontratantes não estabelecidos na UE, quando as atividades estejam relacionadas com a oferta de bens ou serviços ou o controlo do comportamento na UE.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "3", "type": "scope"}
        },
        {
            "question": "Qual é o âmbito de aplicação material do RGPD de acordo com o Artigo 2?",
            "answer": "De acordo com o Artigo 2, RGPD (CELEX: 32016R0679), o Regulamento aplica-se ao tratamento de dados pessoais por meios total ou parcialmente automatizados, bem como ao tratamento por meios não automatizados de dados pessoais contidos num ficheiro ou a ele destinados. Não se aplica ao tratamento efetuado no âmbito de atividades não abrangidas pelo direito da UE, pelos Estados-Membros no âmbito da política externa e de segurança comum, por pessoas singulares no exercício de atividades exclusivamente pessoais ou domésticas, ou pelas autoridades competentes para efeitos de prevenção ou investigação de infrações penais.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "2", "type": "scope"}
        },
        {
            "question": "A quem se aplica o direito ao esquecimento de acordo com o Artigo 17?",
            "answer": "De acordo com o Artigo 17, RGPD (CELEX: 32016R0679), o titular dos dados tem o direito de obter do responsável pelo tratamento o apagamento dos seus dados pessoais sem demora injustificada quando: os dados deixaram de ser necessários, o consentimento foi retirado, o titular se opõe ao tratamento, os dados foram tratados ilicitamente, o apagamento é necessário para cumprimento de obrigação jurídica, ou os dados foram recolhidos no contexto de serviços da sociedade da informação oferecidos a crianças.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "17", "type": "scope"}
        },
        {
            "question": "Que direitos o Artigo 15 confere aos titulares dos dados?",
            "answer": "De acordo com o Artigo 15, RGPD (CELEX: 32016R0679), o titular dos dados tem o direito de obter do responsável pelo tratamento a confirmação de que os dados pessoais que lhe dizem respeito são ou não objeto de tratamento e, em caso afirmativo, acesso aos dados e informações sobre as finalidades do tratamento, as categorias de dados, os destinatários, o prazo de conservação, os direitos de retificação ou apagamento, o direito de apresentar reclamação, a origem dos dados, e a existência de decisões automatizadas.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "15", "type": "scope"}
        },
        {
            "question": "A que se refere CELEX 32016R0679?",
            "answer": "CELEX: 32016R0679 refere-se ao Regulamento (UE) 2016/679 do Parlamento Europeu e do Conselho relativo à proteção das pessoas singulares no que diz respeito ao tratamento de dados pessoais e à livre circulação desses dados (Regulamento Geral sobre a Proteção de Dados - RGPD), adotado em 27 de abril de 2016 e aplicável desde 25 de maio de 2018.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "", "type": "citation"}
        },
        {
            "question": "Que coimas podem ser aplicadas de acordo com o Artigo 83 do RGPD?",
            "answer": "De acordo com o Artigo 83, RGPD (CELEX: 32016R0679), podem ser aplicadas coimas administrativas até 10.000.000 EUR ou 2% do volume de negócios anual mundial total para certas infrações, e coimas até 20.000.000 EUR ou 4% do volume de negócios anual mundial total para as infrações mais graves, como a violação dos princípios básicos do tratamento, das condições de consentimento ou dos direitos dos titulares dos dados.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "83", "type": "citation"}
        },
        {
            "question": "O que o Artigo 13 do RGPD exige que os responsáveis forneçam aos titulares dos dados?",
            "answer": "De acordo com o Artigo 13, RGPD (CELEX: 32016R0679), quando os dados pessoais são recolhidos junto do titular, o responsável deve fornecer: a identidade e contactos do responsável, as finalidades e base jurídica do tratamento, os destinatários dos dados, o prazo de conservação, a existência de direitos, o direito de retirar o consentimento, o direito de apresentar reclamação, se a comunicação de dados é obrigatória, e a existência de decisões automatizadas.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "13", "type": "citation"}
        },
        {
            "question": "Qual é a função do encarregado da proteção de dados de acordo com o Artigo 39 do RGPD?",
            "answer": "De acordo com o Artigo 39, RGPD (CELEX: 32016R0679), o encarregado da proteção de dados: informa e aconselha o responsável, o subcontratante e os trabalhadores sobre as obrigações de proteção de dados; controla a conformidade com o Regulamento; presta aconselhamento sobre as avaliações de impacto; coopera com a autoridade de controlo; e atua como ponto de contacto da autoridade de controlo sobre questões relacionadas com o tratamento.",
            "language": "pt",
            "metadata": {"celex": "32016R0679", "article": "39", "type": "citation"}
        }
    ]
    return questions


def generate_test_set(output_file: str, questions_per_language: int = 20):
    """Generate complete test set with all languages"""
    print(f"Generating test set: {questions_per_language} questions per language")

    # Create all questions
    all_questions = []
    all_questions.extend(create_english_questions())
    all_questions.extend(create_french_questions())
    all_questions.extend(create_german_questions())
    all_questions.extend(create_spanish_questions())
    all_questions.extend(create_portuguese_questions())

    # Verify counts
    lang_counts = {}
    for q in all_questions:
        lang = q['language']
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    print("\nQuestion counts by language:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang.upper()}: {count}")

    print(f"\nTotal questions: {len(all_questions)}")

    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for question in all_questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')

    print(f"\nTest set saved to: {output_file}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description='Generate EUR-Lex Q&A test set')
    parser.add_argument('--output_file', type=str, default='data/test/test_qna_100.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--questions_per_language', type=int, default=20,
                       help='Number of questions per language')

    args = parser.parse_args()

    generate_test_set(args.output_file, args.questions_per_language)
    print("\n✓ Test set generation complete!")


if __name__ == '__main__':
    main()
