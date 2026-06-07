import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DATA_DIR = Path("2026")
OUTPUT_FILE = Path("data/iclr2026_index.json")


def get_value(content, key, default=None):
    item = content.get(key, default)
    if isinstance(item, dict) and "value" in item:
        return item.get("value", default)
    return item


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        parts = re.split(r"[,;]\s*", value)
        return [x.strip() for x in parts if x.strip()]
    return [str(value)]


def clean_text(value, limit=None):
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\|[:\-\s|]*\|", " ", text)
    if limit and len(text) > limit:
        return text[: limit - 1].rstrip() + "..."
    return text


def parse_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def find_source_file(data_dir, mode):
    candidates = sorted(data_dir.glob("*/submissions.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No submissions.jsonl found under {data_dir}")

    if mode == "small":
        return min(candidates, key=lambda p: p.stat().st_size)
    if mode == "reviews":
        return max(candidates, key=lambda p: p.stat().st_size)

    # Auto uses the richer file when available, because it contains reviews and decisions.
    return max(candidates, key=lambda p: p.stat().st_size)


def detect_status(record, content):
    status_views = [str(x).lower() for x in record.get("status_views", [])]
    venue = clean_text(get_value(content, "venue", "")).lower()
    if any("accepted" in x for x in status_views) or "poster" in venue or "spotlight" in venue or "oral" in venue:
        return "accepted"
    if any("withdrawn" in x for x in status_views):
        return "withdrawn"
    if any("desk" in x for x in status_views):
        return "desk_rejected"
    return status_views[0] if status_views else "unknown"


def is_reply_type(reply, name):
    name = name.lower()
    invitations = " ".join(reply.get("invitations", []))
    parent = str(reply.get("parentInvitations", ""))
    haystack = f"{invitations} {parent}".lower()
    return name in haystack


def extract_reviews(record):
    note = record.get("note", {}) or {}
    replies = (
        record.get("details", {}).get("replies")
        or note.get("details", {}).get("replies")
        or []
    )
    official_reviews = []
    decision = ""
    meta_review = ""
    comments = []

    for reply in replies:
        content = reply.get("content", {}) or {}
        if is_reply_type(reply, "decision"):
            decision = clean_text(get_value(content, "decision", "") or get_value(content, "title", ""))
        elif is_reply_type(reply, "meta_review"):
            meta_review = clean_text(
                get_value(content, "summary", "")
                or get_value(content, "recommendation", "")
                or get_value(content, "comment", ""),
                1200,
            )
        elif is_reply_type(reply, "official_review"):
            official_reviews.append(content)
        elif is_reply_type(reply, "official_comment"):
            comment = clean_text(get_value(content, "comment", ""), 500)
            if comment:
                comments.append(comment)

    ratings = []
    confidences = []
    strengths = []
    weaknesses = []
    questions = []
    summaries = []

    for content in official_reviews:
        rating = parse_number(get_value(content, "rating"))
        confidence = parse_number(get_value(content, "confidence"))
        if rating is not None:
            ratings.append(rating)
        if confidence is not None:
            confidences.append(confidence)
        for field, target, limit in [
            ("summary", summaries, 550),
            ("strengths", strengths, 550),
            ("weaknesses", weaknesses, 650),
            ("questions", questions, 450),
        ]:
            text = clean_text(get_value(content, field, ""), limit)
            if text:
                target.append(text)

    return {
        "decision": decision,
        "meta_review": meta_review,
        "review_count": len(official_reviews),
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "avg_confidence": round(sum(confidences) / len(confidences), 2) if confidences else None,
        "summaries": summaries[:3],
        "strengths": strengths[:3],
        "weaknesses": weaknesses[:3],
        "questions": questions[:3],
        "comments": comments[:2],
    }


def extract_record(record, include_reviews):
    note = record.get("note", {}) or {}
    content = note.get("content", {}) or {}
    paper_id = note.get("id") or note.get("forum") or record.get("id")

    title = clean_text(get_value(content, "title", "Untitled"))
    authors = as_list(get_value(content, "authors", []))
    authorids = as_list(get_value(content, "authorids", []))
    keywords = as_list(get_value(content, "keywords", []))
    abstract = clean_text(get_value(content, "abstract", ""), 2600)
    area = clean_text(get_value(content, "primary_area", ""))
    venue = clean_text(get_value(content, "venue", ""))
    pdf = clean_text(get_value(content, "pdf", ""))
    bibtex = clean_text(get_value(content, "_bibtex", ""), 1200)
    status = detect_status(record, content)

    review_data = extract_reviews(record) if include_reviews else {
        "decision": "",
        "meta_review": "",
        "review_count": 0,
        "avg_rating": None,
        "avg_confidence": None,
        "summaries": [],
        "strengths": [],
        "weaknesses": [],
        "questions": [],
        "comments": [],
    }

    search_parts = [
        title,
        " ".join(authors),
        " ".join(keywords),
        area,
        venue,
        abstract,
        review_data.get("meta_review", ""),
        " ".join(review_data.get("summaries", [])),
        " ".join(review_data.get("strengths", [])),
        " ".join(review_data.get("weaknesses", [])),
        " ".join(review_data.get("questions", [])),
        " ".join(review_data.get("comments", [])),
    ]

    return {
        "id": paper_id,
        "forum": note.get("forum", paper_id),
        "title": title,
        "authors": authors,
        "authorids": authorids,
        "keywords": keywords,
        "abstract": abstract,
        "primary_area": area,
        "venue": venue,
        "status": status,
        "pdf": pdf,
        "bibtex": bibtex,
        "openreview_url": f"https://openreview.net/forum?id={paper_id}" if paper_id else "",
        "reviews": review_data,
        "search_text": clean_text(" ".join(x for x in search_parts if x), 7000),
    }


def build_stats(papers):
    areas = Counter(p.get("primary_area") or "unknown" for p in papers)
    keywords = Counter()
    authors = Counter()
    statuses = Counter(p.get("status") or "unknown" for p in papers)
    ratings = []

    for paper in papers:
        keywords.update(k.lower() for k in paper.get("keywords", []) if k)
        authors.update(paper.get("authors", []))
        rating = paper.get("reviews", {}).get("avg_rating")
        if isinstance(rating, (int, float)):
            ratings.append(float(rating))

    return {
        "paper_count": len(papers),
        "status_counts": dict(statuses.most_common()),
        "top_areas": areas.most_common(20),
        "top_keywords": keywords.most_common(40),
        "top_authors": authors.most_common(20),
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "papers_with_reviews": sum(1 for p in papers if p.get("reviews", {}).get("review_count", 0) > 0),
    }


def build_index(args):
    source_file = find_source_file(args.data_dir, args.source)
    include_reviews = args.source != "small"
    selected_status = args.status.lower()
    papers = []
    seen = set()
    scanned = 0
    kept = 0

    with source_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            scanned += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"skip malformed line {scanned}: {exc}")
                continue

            paper = extract_record(record, include_reviews)
            if not paper.get("id") or paper["id"] in seen:
                continue
            if selected_status != "all" and paper.get("status") != selected_status:
                continue

            seen.add(paper["id"])
            papers.append(paper)
            kept += 1
            if args.limit and kept >= args.limit:
                break
            if scanned % 1000 == 0:
                print(f"scanned {scanned}, kept {kept}")

    stats = build_stats(papers)
    payload = {
        "metadata": {
            "name": "ICLR 2026 Paper QA Index",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(source_file),
            "source_mode": args.source,
            "include_reviews": include_reviews,
            "status_filter": selected_status,
            "scanned_records": scanned,
        },
        "stats": stats,
        "papers": papers,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"wrote {args.out} ({size_mb:.1f} MB)")
    print(f"papers: {stats['paper_count']}, with reviews: {stats['papers_with_reviews']}")
    if stats.get("top_areas"):
        print("top areas:")
        for area, count in stats["top_areas"][:8]:
            print(f"  {area}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Build a compact ICLR 2026 paper QA index.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_FILE)
    parser.add_argument(
        "--source",
        choices=["auto", "reviews", "small"],
        default="auto",
        help="auto/reviews uses the richer JSONL with reviews; small uses the lighter paper-only JSONL.",
    )
    parser.add_argument(
        "--status",
        default="accepted",
        help="accepted, withdrawn, desk_rejected, unknown, or all. Default keeps accepted papers for a top-conference demo.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max papers for a tiny demo index.")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    build_index(args)


if __name__ == "__main__":
    main()
