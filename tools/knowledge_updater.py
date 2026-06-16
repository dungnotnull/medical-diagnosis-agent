"""Research paper crawler: PubMed, Cochrane, WHO, MedRxiv → SECOND-KNOWLEDGE-BRAIN.md."""
from __future__ import annotations

import hashlib
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

BRAIN_PATH = Path("SECOND-KNOWLEDGE-BRAIN.md")
LOG_SECTION = "## Knowledge Update Log"

ARXIV_QUERIES = [
    "medical diagnosis artificial intelligence language model",
    "clinical decision support deep learning",
    "triage early warning score machine learning",
    "differential diagnosis natural language processing",
    "medical large language model safety evaluation",
]

PUBMED_QUERIES = [
    "clinical decision support system AI",
    "early warning score triage prediction",
    "natural language processing symptom extraction",
    "ICD diagnosis machine learning",
    "large language model clinical safety",
]

DOMAIN_KEYWORDS = [
    "triage", "diagnosis", "clinical", "symptom", "disease", "patient",
    "medical", "treatment", "emergency", "hospital", "NLP", "deep learning",
    "language model", "ICD", "NEWS2", "qSOFA", "CURB-65", "EHR",
]


@dataclass
class PaperEntry:
    title: str
    authors: str
    year: str
    venue: str
    url: str
    abstract: str
    key_finding: str
    relevance: str

    def identifier(self) -> str:
        return self.url or self.title

    def to_table_row(self) -> str:
        return (
            f"| {self.title[:80]} | {self.authors[:40]} | {self.year} | "
            f"{self.venue[:30]} | {self.url[:60]} | {self.key_finding[:120]} | {self.relevance[:60]} |"
        )


class KnowledgeUpdater:
    def __init__(self, brain_path: str = "SECOND-KNOWLEDGE-BRAIN.md", memory_manager=None):
        self._brain_path = Path(brain_path)
        self._memory = memory_manager

    async def run_update(self) -> dict:
        papers: list[PaperEntry] = []
        papers += self._crawl_pubmed()
        papers += self._crawl_arxiv()
        papers += self._crawl_medrxiv()

        unique = self._deduplicate(papers)
        scored = sorted(unique, key=lambda p: self._score_paper(p), reverse=True)
        top = scored[:30]

        new_count = self._append_to_brain(top)
        result = {
            "new_entries": new_count,
            "total_fetched": len(papers),
            "after_dedup": len(unique),
            "message": f"Added {new_count} new papers to SECOND-KNOWLEDGE-BRAIN.md",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Knowledge update: %s", result["message"])
        return result

    def _crawl_pubmed(self) -> list[PaperEntry]:
        papers = []
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        for query in PUBMED_QUERIES:
            try:
                time.sleep(0.5)
                encoded = urllib.parse.quote(query)
                search_url = f"{base}esearch.fcgi?db=pubmed&term={encoded}&retmax=8&sort=relevance&retmode=xml"
                with urllib.request.urlopen(search_url, timeout=15) as resp:
                    tree = ET.fromstring(resp.read())
                pmids = [el.text for el in tree.findall(".//Id") if el.text]
                if not pmids:
                    continue
                time.sleep(0.5)
                ids_str = ",".join(pmids[:6])
                fetch_url = f"{base}efetch.fcgi?db=pubmed&id={ids_str}&retmode=xml&rettype=abstract"
                with urllib.request.urlopen(fetch_url, timeout=20) as resp:
                    fetch_tree = ET.fromstring(resp.read())
                for article in fetch_tree.findall(".//PubmedArticle"):
                    paper = self._parse_pubmed_article(article)
                    if paper:
                        papers.append(paper)
            except Exception as e:
                logger.warning("PubMed crawl failed for '%s': %s", query, e)
        return papers

    def _parse_pubmed_article(self, article: ET.Element) -> Optional[PaperEntry]:
        try:
            title_el = article.find(".//ArticleTitle")
            title = title_el.text or "Unknown" if title_el is not None else "Unknown"
            abstract_el = article.find(".//AbstractText")
            abstract = abstract_el.text or "" if abstract_el is not None else ""
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text or "" if pmid_el is not None else ""
            year_el = article.find(".//PubDate/Year")
            year = year_el.text or "2024" if year_el is not None else "2024"
            journal_el = article.find(".//Journal/Title")
            venue = journal_el.text or "PubMed" if journal_el is not None else "PubMed"
            authors = []
            for author in article.findall(".//Author")[:3]:
                ln = author.find("LastName")
                fn = author.find("ForeName")
                if ln is not None:
                    authors.append(f"{ln.text} {fn.text[0] if fn is not None and fn.text else ''}")
            return PaperEntry(
                title=title[:100],
                authors=", ".join(authors) or "et al.",
                year=year,
                venue=venue[:50],
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                abstract=abstract[:500],
                key_finding=abstract[:150] if abstract else "See abstract",
                relevance="Clinical AI / Medical NLP",
            )
        except Exception as e:
            logger.debug("Failed to parse PubMed article: %s", e)
            return None

    def _crawl_arxiv(self) -> list[PaperEntry]:
        papers = []
        base = "http://export.arxiv.org/api/query"
        categories = "cs.AI+cs.LG+cs.CL"
        for query in ARXIV_QUERIES[:3]:
            try:
                time.sleep(1)
                encoded = urllib.parse.quote(query)
                url = f"{base}?search_query=all:{encoded}+AND+cat:{categories}&max_results=6&sortBy=submittedDate&sortOrder=descending"
                with urllib.request.urlopen(url, timeout=20) as resp:
                    tree = ET.fromstring(resp.read())
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in tree.findall("atom:entry", ns):
                    paper = self._parse_arxiv_entry(entry, ns)
                    if paper:
                        papers.append(paper)
            except Exception as e:
                logger.warning("ArXiv crawl failed for '%s': %s", query, e)
        return papers

    def _parse_arxiv_entry(self, entry: ET.Element, ns: dict) -> Optional[PaperEntry]:
        try:
            title = (entry.find("atom:title", ns).text or "").strip().replace("\n", " ")
            abstract = (entry.find("atom:summary", ns).text or "").strip()[:500]
            url_el = entry.find("atom:id", ns)
            url = url_el.text.strip() if url_el is not None else ""
            published = (entry.find("atom:published", ns).text or "")[:4]
            authors = []
            for author in entry.findall("atom:author", ns)[:3]:
                name = author.find("atom:name", ns)
                if name is not None:
                    authors.append(name.text)
            return PaperEntry(
                title=title[:100],
                authors=", ".join(authors) or "et al.",
                year=published,
                venue="arXiv",
                url=url[:80],
                abstract=abstract,
                key_finding=abstract[:150] if abstract else "See abstract",
                relevance="AI/ML Medical Research",
            )
        except Exception as e:
            logger.debug("ArXiv parse error: %s", e)
            return None

    def _crawl_medrxiv(self) -> list[PaperEntry]:
        papers = []
        try:
            rss_url = "https://www.medrxiv.org/rss/medrxiv.xml"
            with urllib.request.urlopen(rss_url, timeout=20) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            root = ET.fromstring(content)
            for item in root.findall(".//item")[:15]:
                title_el = item.find("title")
                desc_el = item.find("description")
                link_el = item.find("link")
                pub_date_el = item.find("pubDate")
                title = title_el.text or "Unknown" if title_el is not None else "Unknown"
                abstract = desc_el.text or "" if desc_el is not None else ""
                url = link_el.text or "" if link_el is not None else ""
                year = (pub_date_el.text or "2024")[:4] if pub_date_el is not None else "2024"
                if any(kw.lower() in title.lower() or kw.lower() in abstract.lower()
                       for kw in ["diagnosis", "triage", "clinical", "AI", "NLP", "language model"]):
                    papers.append(PaperEntry(
                        title=title[:100],
                        authors="MedRxiv Authors",
                        year=year,
                        venue="MedRxiv",
                        url=url[:80],
                        abstract=abstract[:300],
                        key_finding=abstract[:150] if abstract else "See preprint",
                        relevance="Medical Preprint",
                    ))
        except Exception as e:
            logger.warning("MedRxiv crawl failed: %s", e)
        return papers

    def _deduplicate(self, papers: list[PaperEntry]) -> list[PaperEntry]:
        seen: set[str] = set()
        unique: list[PaperEntry] = []
        for p in papers:
            h = hashlib.sha256(p.identifier().encode()).hexdigest()
            if self._memory and self._memory.is_known_paper(p.identifier()):
                continue
            if h not in seen:
                seen.add(h)
                unique.append(p)
        return unique

    def _score_paper(self, paper: PaperEntry) -> float:
        try:
            year = int(paper.year)
        except ValueError:
            year = 2020
        current_year = datetime.now().year
        recency = max(0.0, 1.0 - (current_year - year) / 5.0)

        text = f"{paper.title} {paper.abstract}".lower()
        keyword_hits = sum(1 for kw in DOMAIN_KEYWORDS if kw.lower() in text)
        relevance = min(1.0, keyword_hits / 5.0)

        return 0.6 * recency + 0.4 * relevance

    def _append_to_brain(self, papers: list[PaperEntry]) -> int:
        if not self._brain_path.exists():
            logger.warning("SECOND-KNOWLEDGE-BRAIN.md not found — skipping append")
            return 0

        content = self._brain_path.read_text(encoding="utf-8")
        new_count = 0
        new_rows = []

        for paper in papers:
            if not self._is_in_brain(content, paper):
                new_rows.append(paper.to_table_row())
                if self._memory:
                    self._memory.mark_paper_known(paper.identifier(), paper.url, paper.title)
                new_count += 1

        if new_rows:
            table_header = "| Title | Authors | Year | Venue | DOI/Link | Key Finding | Relevance |"
            separator = "|-------|---------|------|-------|----------|-------------|-----------|"
            if table_header not in content:
                insert_after = "## Key Research Papers"
                if insert_after in content:
                    rows_block = f"\n\n{table_header}\n{separator}\n" + "\n".join(new_rows) + "\n"
                    content = content.replace(insert_after, insert_after + rows_block, 1)
                else:
                    content += f"\n\n{table_header}\n{separator}\n" + "\n".join(new_rows) + "\n"
            else:
                content += "\n" + "\n".join(new_rows)

        log_entry = (
            f"| {datetime.now(timezone.utc).strftime('%Y-%m-%d')} "
            f"| PubMed+ArXiv+MedRxiv | {new_count} | — | Auto-crawl |"
        )
        if LOG_SECTION in content:
            content = content.replace(LOG_SECTION, LOG_SECTION + f"\n{log_entry}")
        else:
            content += f"\n\n{LOG_SECTION}\n\n| Date | Source | New Entries | Total | Notes |\n|------|--------|-------------|-------|-------|\n{log_entry}\n"

        self._brain_path.write_text(content, encoding="utf-8")
        return new_count

    def _is_in_brain(self, content: str, paper: PaperEntry) -> bool:
        check = paper.url[:40] if paper.url else paper.title[:40]
        return check in content


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    updater = KnowledgeUpdater()
    result = asyncio.run(updater.run_update())
    print(result)
