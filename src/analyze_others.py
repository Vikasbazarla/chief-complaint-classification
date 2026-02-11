# src/analyze_others.py
"""
analyze_others.py

Usage:
    python src/analyze_others.py

What it does:
- Reads data/labeled_sample.csv
- Finds rows where Primary Classification == "Other"
- Aggregates unique processed complaint strings and counts
- Suggests a label for each unique string by checking:
    1) substring match against keywords in data/rules.json
    2) fuzzy matching (RapidFuzz) against keywords (fallback)
- Outputs data/other_mappings.csv for manual review and editing
- Outputs data/other_samples_for_review.csv containing a handful of raw examples per processed string
"""

import os
import json
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz, process

BASE = os.path.join(os.path.dirname(__file__), "..")
LABELED_PATH = os.path.join(BASE, "data", "labeled_sample.csv")
RULES_PATH = os.path.join(BASE, "data", "rules.json")
OUT_PATH = os.path.join(BASE, "data", "other_mappings.csv")
SAMPLE_OUT = os.path.join(BASE, "data", "other_samples_for_review.csv")

# Parameters (tune if needed)
FUZZY_THRESHOLD = 82  # if fuzzy match score >= threshold, consider as candidate
MAX_EXAMPLES_PER_PROCESSED = 5

def load_rules(path=RULES_PATH):
    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    # Build reverse lookup: keyword -> label
    keyword_to_label = {}
    # We'll also keep a precompiled list of (label, keyword) pairs for fuzzy search
    pairs = []
    for label, keywords in rules.items():
        for kw in keywords:
            kw_clean = str(kw).lower().strip()
            keyword_to_label[kw_clean] = label
            pairs.append((label, kw_clean))
    return rules, keyword_to_label, pairs

def suggest_label_for_text(text, rules_pairs, keyword_to_label):
    """
    text: processed complaint (lowercased)
    rules_pairs: list of (label, keyword)
    keyword_to_label: dict keyword->label
    returns: (suggested_label, best_keyword, score, method)
    """
    t = str(text).lower().strip()

    # 1) Exact substring matching (prefer longer keyword matches)
    # We'll look for keywords that appear as whole word or substring
    candidate_matches = []
    for label, kw in rules_pairs:
        if not kw:
            continue
        if kw in t:
            # give score based on length of keyword (longer is better)
            score = 100 + len(kw)
            candidate_matches.append((label, kw, score, "substr"))
    if candidate_matches:
        # choose highest score (i.e., longest matching keyword)
        candidate_matches.sort(key=lambda x: (-x[2], x[1]))
        label, kw, score, method = candidate_matches[0]
        return label, kw, score, method

    # 2) Token-level whole-word matching (split t into tokens and check exact token match)
    tokens = t.split()
    for label, kw in rules_pairs:
        if kw in tokens:
            return label, kw, 100, "token"

    # 3) Fuzzy fallback: match against all rule keywords using partial_ratio
    # Use rapidfuzz process.extract to get best matches quickly
    # Build a list of unique keywords
    unique_keywords = list({kw for _, kw in rules_pairs if kw})
    # process.extract returns tuples (match, score, index)
    best = process.extractOne(t, unique_keywords, scorer=fuzz.partial_ratio)
    if best:
        matched_kw, score, _ = best
        if score >= FUZZY_THRESHOLD:
            label = keyword_to_label.get(matched_kw, None)
            if label:
                return label, matched_kw, score, "fuzzy"
    # 4) No good match
    return "Other", "", 0, "none"

def main():
    if not os.path.exists(LABELED_PATH):
        print("Error: labeled_sample.csv not found at", LABELED_PATH)
        return
    if not os.path.exists(RULES_PATH):
        print("Error: rules.json not found at", RULES_PATH)
        return

    print("Loading labeled data...")
    df = pd.read_csv(LABELED_PATH, dtype=str, keep_default_na=False)

    # normalize columns
    if "Primary Classification" not in df.columns or "processed" not in df.columns:
        raise ValueError("Expected columns 'processed' and 'Primary Classification' in labeled_sample.csv")

    df['Primary Classification'] = df['Primary Classification'].astype(str).str.strip()

    # filter other rows
    other_df = df[df['Primary Classification'].str.lower() == 'other'].copy()
    if other_df.empty:
        print("No rows labeled 'Other' found. Nothing to analyze.")
        return

    total_other = len(other_df)
    print(f"Found {total_other} rows labeled as 'Other'")

    # load rules
    rules, keyword_to_label, pairs = load_rules()

    # aggregate by processed text
    agg = other_df.groupby('processed').agg({
        'raw': lambda s: list(s.dropna().unique())[:MAX_EXAMPLES_PER_PROCESSED],
        'processed': 'count'
    }).rename(columns={'processed': 'count'})

    # Build mapping results
    results = []
    for proc_text, row in agg.iterrows():
        count = int(row['count'])
        examples = row['raw'] if isinstance(row['raw'], list) else [row['raw']]
        # ensure examples are strings and limit length
        examples = [str(x) for x in examples][:MAX_EXAMPLES_PER_PROCESSED]
        suggested_label, best_kw, score, method = suggest_label_for_text(proc_text, pairs, keyword_to_label)
        results.append({
            'processed': proc_text,
            'count': count,
            'examples': " || ".join(examples),
            'suggested_label': suggested_label,
            'best_match_keyword': best_kw,
            'match_score': score,
            'match_method': method
        })

    # create dataframe and sort by count desc
    res_df = pd.DataFrame(results).sort_values(by='count', ascending=False).reset_index(drop=True)

    # Save CSV for manual review
    res_df.to_csv(OUT_PATH, index=False)
    print("Wrote suggestions to:", OUT_PATH)

    # Also save a sample file with raw examples for quick manual review
    sample_rows = other_df.groupby('processed').head(3)[['raw','processed','Multi-label Classification','Primary Classification']]
    sample_rows.to_csv(SAMPLE_OUT, index=False)
    print("Wrote sample rows to:", SAMPLE_OUT)

    # Print quick stats
    print("\nTop 20 'Other' processed complaints (by count) with suggested labels:")
    display_df = res_df.head(20)[['processed','count','suggested_label','best_match_keyword','match_score','match_method']]
    print(display_df.to_string(index=False))

    print("\nIf the suggested labels look good, you can:")
    print("  - Manually review and edit the CSV:", OUT_PATH)
    print("  - After confirmation, we can auto-merge these mappings into rules.json")

if __name__ == "__main__":
    main()
