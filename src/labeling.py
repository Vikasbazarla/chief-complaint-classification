# src/labeling.py
import os
import json
import re
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz, process

BASE = os.path.join(os.path.dirname(__file__), "..")
RULES_PATH = os.path.join(BASE, "data", "rules.json")
INPUT = os.path.join(BASE, "data", "processed_sample.csv")
OUTPUT = os.path.join(BASE, "data", "labeled_sample.csv")

# --- load rules ---
with open(RULES_PATH, "r", encoding="utf-8") as f:
    rules = json.load(f)

# --- build regex patterns (word boundary aware) ---
pattern_map = {}
for label, kws in rules.items():
    pats = []
    for kw in kws:
        kw_s = str(kw).strip().lower()
        if not kw_s:
            continue
        # escape and allow whitespace/ punctuation boundaries
        kw_clean = re.escape(kw_s)
        pats.append(r'(?:^|\s)'+kw_clean+r'(?:$|\s|[.,!?])')
    if pats:
        pattern_map[label] = re.compile("|".join(pats), flags=re.IGNORECASE)

# --- prepare tokenized keyword cache for token-presence checks ---
rules_token_map = {}
for label, kws in rules.items():
    tokenized = []
    for kw in kws:
        kw_s = str(kw).strip().lower()
        if not kw_s:
            continue
        tokens = [t for t in re.findall(r"[a-z0-9]+", kw_s)]
        if tokens:
            tokenized.append((kw_s, tokens))
    rules_token_map[label] = tokenized

# --- priority order (adjust if you want different primary selection) ---
priority_order = [
    "Fever", "Cardiovascular / Chest Issues", "RTA", "Injury & Trauma", "General Malaise & Systemic",
    "Gynecological", "GIT", "URTI (moderate)", "URTI (mild)", "URTI (other)", "ENT",
    "Migraine", "Headache", "Giddiness", "Anxiety", "Musculoskeletal", "Pain (non Traumatic)",
    "Tooth Ache", "Skin & Dermatological", "Eyes", "Allergy", "Dressing",
    "BP Checkup", "GRBS Checkup", "Test", "General Checkup", "Other"
]
priority_map = {c: i for i, c in enumerate(priority_order)}

# --- helper sets for heuristics ---
WOUND_VERBS = {"cut", "cuts", "cutting", "laceration", "lacerations", "lacerated", "bleed", "bleeding", "stab", "stabbed", "puncture", "abrasion"}
BODY_PARTS = {"finger", "thumb", "toe", "hand", "leg", "arm", "foot", "wrist", "ankle", "knee", "back", "shoulder", "neck", "head", "ear", "eye", "mouth", "tooth"}
ACCIDENT_KEYWORDS = {"rta", "accident", "road", "bike", "car", "collision", "fell", "fall", "hit by", "skid"}

# --- classification function ---
def classify_row(text: str):
    """
    Returns: (labels_list, primary_label)
    """
    if not isinstance(text, str) or not text.strip():
        return ["Other"], "Other"

    t = text.lower().strip()
    labels = []

    # 1) explicit pain handling (preserve domain-specific choices)
    if "pain" in t or "ache" in t:
        if any(x in t for x in ["neck pain", "back pain", "joint pain", "knee pain", "shoulder pain", "hand pain", "leg pain", "ankle pain"]):
            labels.append("Musculoskeletal")
        elif "throat pain" in t or "ear pain" in t:
            labels.append("ENT")
        elif "abdominal pain" in t or "stomach pain" in t or "epigastric" in t:
            labels.append("GIT")
        else:
            labels.append("Pain (non Traumatic)")

    # 2) regex exact-ish matching from rules
    for label, pat in pattern_map.items():
        try:
            if pat.search(t) and label not in labels:
                labels.append(label)
        except re.error:
            # skip problematic regex
            continue

    # 3) token-presence matching for multi-word keywords (robust to word order)
    #    if all tokens of a keyword exist in the processed text (any order), it's a match
    text_tokens = set(re.findall(r"[a-z0-9]+", t))
    if text_tokens:
        for label, tokenized_list in rules_token_map.items():
            # skip if label already added
            if label in labels:
                continue
            for kw_s, kw_tokens in tokenized_list:
                if len(kw_tokens) == 1:
                    # single token handled by regex above, but check as fallback
                    if kw_tokens[0] in text_tokens:
                        labels.append(label)
                        break
                else:
                    # multi-token: require all tokens present
                    if all(tok in text_tokens for tok in kw_tokens):
                        labels.append(label)
                        break
            # next label

    # 4) heuristics: wound verb + body part -> Injury & Trauma
    if not ("Injury & Trauma" in labels):
        if any(w in t for w in WOUND_VERBS) and any(bp in t for bp in BODY_PARTS):
            labels.append("Injury & Trauma")

    # 5) accident / RTA strong signal -> RTA
    if not ("RTA" in labels):
        if any(a in t for a in ACCIDENT_KEYWORDS):
            # small guard: ensure "medical" "fitness" words don't cause RTA
            if not any(x in t for x in ["medical fitness", "medical certificate", "medical checkup", "checkup"]):
                labels.append("RTA")

    # 6) fuzzy fallback when no label found (conservative threshold)
    if not labels:
        # try to find best label by partial fuzzy match across keywords
        best_label = None
        best_score = 0
        for label, kws in rules.items():
            for kw in kws:
                kw_s = str(kw).lower().strip()
                if not kw_s:
                    continue
                score = fuzz.partial_ratio(kw_s, t)
                if score > best_score:
                    best_score = score
                    best_label = label
        # require reasonably high score to accept
        if best_score >= 88 and best_label:
            labels.append(best_label)

    # final fallback: Other
    if not labels:
        labels = ["Other"]

    # dedupe preserve order
    seen = set()
    final_labels = []
    for l in labels:
        if l not in seen:
            final_labels.append(l)
            seen.add(l)

    # choose primary based on priority_order
    primary = sorted(final_labels, key=lambda x: priority_map.get(x, 999))[0]
    return final_labels, primary

# --- runner ---
def run(input_path=INPUT, output_path=OUTPUT, chunk_size=2000):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")
    chunks = pd.read_csv(input_path, chunksize=chunk_size, dtype=str, keep_default_na=False)
    outs = []
    for chunk in tqdm(chunks, desc="Labeling"):
        if 'processed' not in chunk.columns:
            raise ValueError("'processed' column missing from processed CSV")
        chunk['processed'] = chunk['processed'].astype(str)
        ml, prim = zip(*chunk['processed'].apply(classify_row))
        chunk['Multi-label Classification'] = [", ".join(x) for x in ml]
        chunk['Primary Classification'] = prim
        outs.append(chunk[['raw','processed','Multi-label Classification','Primary Classification']])
    df_final = pd.concat(outs, ignore_index=True)
    df_final.to_csv(output_path, index=False)
    print("✅ Labeled file saved to:", output_path)
    print("Total rows processed:", len(df_final))

if __name__ == "__main__":
    run()
