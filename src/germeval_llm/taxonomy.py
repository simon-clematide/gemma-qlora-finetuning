"""Official GermEval 2017 taxonomy definitions, label inventories, and typo patch utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Frozen canonical order of the 19 standard GermEval 2017 main aspect categories
MAIN_CATEGORIES_19: tuple[str, ...] = (
    "Allgemein",
    "Atmosphäre",
    "Auslastung_und_Platzangebot",
    "Barrierefreiheit",
    "Connectivity",
    "DB_App_und_Website",
    "Design",
    "Gastronomisches_Angebot",
    "Gepäck",
    "Image",
    "Informationen",
    "Komfort_und_Ausstattung",
    "Reisen_mit_Kindern",
    "Service_und_Kundenbetreuung",
    "Sicherheit",
    "Sonstige_Unregelmässigkeiten",
    "Ticketkauf",
    "Toiletten",
    "Zugfahrt",
)

# Rare split-specific category present once in dev_v1.4.xml and once in test_TIMESTAMP1.xml (absent in train_v1.4.xml)
RARE_UNSEEN_CATEGORIES: tuple[str, ...] = ("QR-Code",)

POLARITIES: tuple[str, ...] = ("positive", "neutral", "negative")

# Official concise German category guidelines for P1/P2/P3 and fine-tuning instructions
CATEGORY_GUIDELINES_DE: dict[str, str] = {
    "Allgemein": "Allgemeine Aussagen über die Deutsche Bahn, das Bahnfahren oder das Unternehmen ohne spezifischeren Unteraspekt.",
    "Atmosphäre": "Stimmung, Ruhe, Lärm, Verhalten anderer Fahrgäste oder Atmosphäre im Zug/Bahnhof.",
    "Auslastung_und_Platzangebot": "Füllgrad der Züge, Sitzplatzverfügbarkeit, Überfüllung, Stehplätze oder Platzangebot.",
    "Barrierefreiheit": "Zugänglichkeit für mobilitätseingeschränkte Reisende, Rollstuhlplätze, Aufzüge, Rampen, Einstiegshilfen.",
    "Connectivity": "WLAN, Internetverbindung, Mobilfunkempfang und Netzabdeckung im Zug oder Bahnhof.",
    "DB_App_und_Website": "Digitale Kanäle wie DB Navigator App, bahn.de Website, Online-Funktionen und digitale Buchung.",
    "Design": "Optisches Erscheinungsbild, Gestaltung, Architektur oder Design von Zügen, Bahnhöfen und Produkten.",
    "Gastronomisches_Angebot": "Bordbistro, Bordrestaurant, Speisen, Getränke und Verpflegung im Zug oder Bahnhof.",
    "Gepäck": "Gepäckablagen, Koffertransport, Fahrradmitnahme oder Gepäckaufbewahrung.",
    "Image": "Öffentliches Ansehen, Ruf, PR-Aktionen, Management, Streiks oder Unternehmenspolitik der Deutschen Bahn.",
    "Informationen": "Fahrgastinformation, Durchsagen, Anzeigetafeln, Verspätungsmeldungen und Auskunft bei Störungen.",
    "Komfort_und_Ausstattung": "Sitzkomfort, Klimaanlage, Heizung, Steckdosen, Sauberkeit (außer WC) und technische Ausstattung der Wagen.",
    "Reisen_mit_Kindern": "Familienfreundlichkeit, Kleinkindabteile, Kinderbetreuung, Kinderwagenstellplätze.",
    "Service_und_Kundenbetreuung": "Freundlichkeit, Hilfsbereitschaft und Kompetenz des Personals, Zugbegleiter, Kundendialog, Entschädigung/Kulanz.",
    "Sicherheit": "Sicherheitsgefühl, Bundespolizei, Übergriffe, Notfälle, technische Sicherheit und Überwachung.",
    "Sonstige_Unregelmässigkeiten": "Streckensperrungen, Polizeieinsätze, Unwetter, Personen im Gleis, Weichen-/Stellwerksstörungen außerhalb des regulären Zuglaufs.",
    "Ticketkauf": "Fahrkartenpreise, Tarife, Sparpreise, BahnCard, Fahrkartenautomaten, Reisezentrum und Ticketkontrolle.",
    "Toiletten": "Verfügbarkeit, Funktion und Sauberkeit der WCs in Zügen und Bahnhöfen.",
    "Zugfahrt": "Pünktlichkeit, Verspätungen, Zugausfälle, Anschlüsse, Fahrzeit, Wagenreihung und konkreter Fahrtverlauf.",
}

# Known gold XML typos documented in Section 2.2
KNOWN_POLARITY_TYPOS: dict[str, str] = {
    "positve": "positive",
    " negative": "negative",
}

KNOWN_CATEGORY_FULL_TYPOS: dict[str, str] = {
    "Sichaupt": "Sicherheit#Haupt",
}

KNOWN_CATEGORY_MAIN_TYPOS: dict[str, str] = {
    "Sichaupt": "Sicherheit",
}


def get_allowed_task_c_labels(include_qr_code: bool = False) -> list[str]:
    """Return deterministically ordered 'Category:polarity' label inventory."""
    cats = list(MAIN_CATEGORIES_19)
    if include_qr_code:
        cats.extend(RARE_UNSEEN_CATEGORIES)
    labels: list[str] = []
    for cat in cats:
        for pol in POLARITIES:
            labels.append(f"{cat}:{pol}")
    return labels


def sort_task_c_labels(labels: list[str] | set[str]) -> list[str]:
    """Sort 'Category:polarity' labels by canonical taxonomy order then polarity order."""
    cat_rank = {c: i for i, c in enumerate(MAIN_CATEGORIES_19 + RARE_UNSEEN_CATEGORIES)}
    pol_rank = {p: i for i, p in enumerate(POLARITIES)}

    def _key(label: str) -> tuple[int, str, int, str]:
        if ":" in label:
            cat, pol = label.split(":", 1)
        else:
            cat, pol = label, ""
        return (cat_rank.get(cat, 999), cat, pol_rank.get(pol, 999), pol)

    return sorted(set(labels), key=_key)


def format_taxonomy_guidelines() -> str:
    """Format the 19 category descriptions as a bulleted German guideline block."""
    lines = [f"- {cat}: {CATEGORY_GUIDELINES_DE[cat]}" for cat in MAIN_CATEGORIES_19]
    return "\n".join(lines)


def build_taxonomy_manifest() -> dict[str, Any]:
    """Construct the frozen taxonomy manifest dictionary for data/manifests/taxonomy.json."""
    return {
        "taxonomy_version": "v1_19cats",
        "num_main_categories": len(MAIN_CATEGORIES_19),
        "main_categories": list(MAIN_CATEGORIES_19),
        "rare_unseen_in_train_categories": list(RARE_UNSEEN_CATEGORIES),
        "polarities": list(POLARITIES),
        "allowed_labels_57": get_allowed_task_c_labels(include_qr_code=False),
        "category_guidelines_de": CATEGORY_GUIDELINES_DE,
        "known_xml_typos": {
            "polarity": KNOWN_POLARITY_TYPOS,
            "category_full": KNOWN_CATEGORY_FULL_TYPOS,
            "category_main": KNOWN_CATEGORY_MAIN_TYPOS,
        },
    }


def save_taxonomy_manifest(path: str | Path) -> Path:
    """Write taxonomy.json to disk."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(build_taxonomy_manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out
