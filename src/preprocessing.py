# src/preprocessing.py
import os
import json
import re
from symspellpy.symspellpy import SymSpell, Verbosity
from rapidfuzz import fuzz

# ---------- config ----------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DICT_PATH = os.path.join(BASE_DIR, "data", "frequency_dictionary_en_82_765.txt")
MEDICAL_DICT_PATH = os.path.join(BASE_DIR, "data", "medical_terms.txt")
ABBR_PATH = os.path.join(BASE_DIR, "data", "abbreviations.json")
NORMALIZE_PATH = os.path.join(BASE_DIR, "data", "normalization.json")
RULES_PATH = os.path.join(BASE_DIR, "data", "rules.json")
# ----------------------------

# Load SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
try:
    if os.path.exists(DICT_PATH):
        sym_spell.load_dictionary(DICT_PATH, term_index=0, count_index=1, separator="\t")
    if os.path.exists(MEDICAL_DICT_PATH):
        sym_spell.load_dictionary(MEDICAL_DICT_PATH, term_index=0, count_index=1, separator="\t")
except Exception:
    pass

# Load configs
abbreviation_dict = {}
normalization_dict = {}
rules_phrases = set()
try:
    if os.path.exists(ABBR_PATH):
        with open(ABBR_PATH, "r", encoding="utf-8") as f:
            abbreviation_dict = json.load(f)
except Exception:
    abbreviation_dict = {}

try:
    if os.path.exists(NORMALIZE_PATH):
        with open(NORMALIZE_PATH, "r", encoding="utf-8") as f:
            normalization_dict = json.load(f)
except Exception:
    normalization_dict = {}

try:
    if os.path.exists(RULES_PATH):
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            temp_rules = json.load(f)
            for kws in temp_rules.values():
                for kw in kws:
                    rules_phrases.add(re.sub(r'\s+', ' ', str(kw).lower().strip()))
except Exception:
    rules_phrases = set()

STRIP_CHARS = ".,;:-()[]\"'`"

body_part_map = {
    "b/l": "bilateral",
    "bil": "bilateral",
    "rt": "right",
    "lt": "left",
    "lf": "left",
    "rta": "road traffic accident",
}

protected_tokens = {
    "bilateral","right","left","knee","leg","hand","injury","fracture","pain",
    "headache","vomiting","diarrhea","fever","cough","cold","sugar","diabetes",
    "bp","blood","pressure","accident","chest","nose","throat","swelling","edema",
    "runny","abdominal","malaria","breathlessness","sorethroat","sore","heartburn",
    "acidity","ulcer","rash","itching","burning","sensation","colicky","epigastric",
    "upper","episode","muscle","irritation","checkup","stool","stools","loose",
    "mouth","sneeze","hit","by","bike","rta","pt","c/o","ear","mhc","follow","up",
    "high", "flow", "menstrual", "period"
}

medical_corrections = {
    "sneez": "sneeze",
    "fevr": "fever",
    "hedache": "headache",
    "sorethrot": "sorethroat",
    "soretrout": "sorethroat",
    "headake": "headache",
    "vomimg": "vomiting",
    "vomitting": "vomiting",
    "righyt": "right",
    "year": "ear",
    "mepz": "",
    "lass": "loss",
    "lition": "lesion",
    "vission": "vision",
    "distrapance": "disturbance",
    "evaluvation": "evaluation",
    "loss sleep": "sleep disturbance",
    "high flow": "heavy menstrual bleeding",
    "high flow menstrual": "heavy menstrual bleeding",
    "period high flow": "heavy menstrual bleeding"
}

def pre_normalize_raw_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = t.strip('"').strip("'").replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r'\s+', ' ', t).strip()
    tl = t.lower()

    tl = re.sub(r'^(came|comes|cames|come|visited|walked in|walkin|walk-in)\b[\s:\-]*', '', tl)
    tl = re.sub(r'\b(came consult|came consultation|came consult|consult|consultation|consulted)\b', '', tl)
    tl = re.sub(r'\b(follow up advised|follow up|follow-up|followup)\b', 'follow up', tl)

    tl = re.sub(r'\bmedical finess\b', 'medical fitness', tl)
    tl = re.sub(r'\bmedical finess certification\b', 'medical fitness certification', tl)

    tl = re.sub(r'\bbee[\s\-]*bite\b', 'bee bite', tl)
    tl = re.sub(r'\bbee[\s\-]*sting\b', 'bee sting', tl)
    tl = re.sub(r'\binsect[\s\-]*bite\b', 'insect bite', tl)

    tl = re.sub(r'\bout\s*side\b', ' outside', tl)
    tl = re.sub(r'\bdr\s+[a-z]{1,30}\b', '', tl)

    tl = re.sub(r'[^\x00-\x7F]+', ' ', tl)
    tl = re.sub(r'\s+', ' ', tl).strip()

    return tl

def normalize_body_parts(tokens):
    normalized = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        clean_token = token.lower().strip(STRIP_CHARS)
        if clean_token == "b/l" or (clean_token == "b" and i + 1 < len(tokens) and tokens[i + 1].lower().strip(STRIP_CHARS) == "l"):
            normalized.append("bilateral")
            if clean_token == "b":
                i += 2
            else:
                i += 1
        elif clean_token in body_part_map:
            normalized.append(body_part_map[clean_token])
            i += 1
        else:
            normalized.append(token)
            i += 1
    return normalized

def expand_abbreviations_tokens(tokens):
    expanded = []
    for token in tokens:
        clean_token = token.lower().strip(STRIP_CHARS)
        if clean_token == 'c':
            continue
        if clean_token == 'c/o':
            continue
        if clean_token in abbreviation_dict:
            expansion = abbreviation_dict[clean_token]
            if isinstance(expansion, str):
                parts = expansion.split()
                if 1 <= len(parts) <= 4:
                    expanded.extend(parts)
                else:
                    expanded.append(token)
            else:
                expanded.append(token)
        else:
            expanded.append(token)
    return expanded

def apply_normalization_tokens(tokens):
    normalized = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            current = tokens[i].lower().strip(STRIP_CHARS)
            next_t = tokens[i + 1].lower().strip(STRIP_CHARS)
            two_word = f"{current} {next_t}"
            if two_word in normalization_dict:
                normalized.append(normalization_dict[two_word])
                i += 2
                continue
        current = tokens[i].lower().strip(STRIP_CHARS)
        if current in normalization_dict:
            normalized.append(normalization_dict[current])
        else:
            normalized.append(tokens[i])
        i += 1
    return normalized

def correct_spelling(tokens):
    corrected = []
    for token in tokens:
        word = token.lower().strip(STRIP_CHARS)
        if not word:
            continue
        if word in protected_tokens or len(word) <= 2 or word.isdigit():
            corrected.append(word)
            continue
        if word in medical_corrections:
            corr = medical_corrections[word]
            if corr:
                corrected.append(corr)
            continue
        try:
            if sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=0):
                corrected.append(word)
                continue
        except Exception:
            pass
        try:
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
        except Exception:
            suggestions = []
        if suggestions:
            best = suggestions[0]
            best_term = getattr(best, "term", str(best))
            similarity = fuzz.ratio(word, best_term)
            best_distance = getattr(best, "distance", None)
            if similarity >= 75 or (len(word) > 4 and best_distance == 1):
                corrected.append(best_term)
            else:
                corrected.append(word)
        else:
            corrected.append(word)
    return corrected

def split_compound_words(tokens):
    split_tokens = []
    for token in tokens:
        token_l = token.lower()
        if len(token_l) < 6 or token_l in protected_tokens:
            split_tokens.append(token_l)
            continue
        try:
            if sym_spell.lookup(token_l, Verbosity.TOP, max_edit_distance=0):
                split_tokens.append(token_l)
                continue
        except Exception:
            pass
        try:
            segmentation = sym_spell.word_segmentation(token_l)
            dist_sum = getattr(segmentation, "distance_sum", 999)
            if dist_sum < len(token_l) * 0.4:
                parts = segmentation.corrected_string.split()
                if len(parts) > 1 and all(len(p) > 2 for p in parts):
                    split_tokens.extend(parts)
                else:
                    split_tokens.append(token_l)
            else:
                split_tokens.append(token_l)
        except Exception:
            split_tokens.append(token_l)
    return split_tokens

def extract_positive_symptoms(text):
    if not text:
        return ""
    text = text.lower().strip()
    hit_by_bike_preserved = False
    if re.search(r'hit\s+by\s+bike', text):
        hit_by_bike_preserved = True
    but_patterns = [
        r'(?:denies?|no|not|without|negative for)\s+[^,]*?but\s+(?:complains of\s+|c/o\s+|has\s+)?([^,\.]+)',
        r'doesn\'t have\s+[^,]*?but\s+has\s+([^,\.]+)'
    ]
    positive_from_but = []
    for pattern in but_patterns:
        but_matches = re.findall(pattern, text, re.IGNORECASE)
        for match in but_matches:
            positive_from_but.append(match.strip())
    negation_patterns = [
        r'\b(?:denies?|negative for|no|not|without|doesn\'t have|absence of)\b[^,\.]*',
        r'\bpt\b',
        r'\bpatient\b',
        r'\bc/o\b',
        r'\bcomplains? of\b',
        r'\bcomplains?\b',
        r'\bhas\b(?=\s+(?:been|a|an|the))',
        r'\bsince\s+\d+\s+days?\b',
        r'\byesterday\b',
        r'\bmild\b'
    ]
    cleaned = text
    for pattern in negation_patterns:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
    if positive_from_but:
        cleaned += ' ' + ' '.join(positive_from_but)
    if hit_by_bike_preserved and "hit by bike" not in cleaned:
        cleaned += ' hit by bike'
    cleaned = re.sub(r'[,/()]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def post_cleanup_tokens(tokens):
    cleaned = []
    for i, tok in enumerate(tokens):
        t = tok.strip(STRIP_CHARS).lower()
        if t.startswith('o') and len(t) > 3:
            candidate = t[1:]
            if i + 1 < len(tokens):
                nxt = tokens[i+1].lower().strip(STRIP_CHARS)
                if nxt in {'nose','cough','sneeze','headache','pain','throat'}:
                    t = candidate
        cleaned.append(t)
    return [c for c in cleaned if c]

def merge_known_phrases(tokens, phrase_set):
    if not phrase_set:
        return tokens
    phrase_list = sorted(phrase_set, key=lambda x: -len(x.split()))
    i = 0
    merged = []
    lower_tokens = [t.lower().strip(STRIP_CHARS) for t in tokens]
    while i < len(tokens):
        matched = False
        for phrase in phrase_list:
            words = phrase.split()
            L = len(words)
            if i + L <= len(tokens) and lower_tokens[i:i+L] == words:
                merged.append(" ".join(tokens[i:i+L]))
                i += L
                matched = True
                break
        if not matched:
            merged.append(tokens[i])
            i += 1
    return merged

def clean_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    text = pre_normalize_raw_text(text)
    if not text:
        return ""
    text = extract_positive_symptoms(text)
    if not text:
        return ""
    tokens = re.findall(r"[A-Za-z0-9\-']+", text)
    if not tokens:
        return ""
    tokens = normalize_body_parts(tokens)
    tokens = expand_abbreviations_tokens(tokens)
    tokens = apply_normalization_tokens(tokens)
    tokens = [t.strip(STRIP_CHARS) for t in tokens if t and t.strip(STRIP_CHARS)]
    tokens = correct_spelling(tokens)
    tokens = split_compound_words(tokens)
    tokens = merge_known_phrases(tokens, rules_phrases)
    tokens = post_cleanup_tokens(tokens)

    stop_words = {
        'a','an','the','of','in','on','at','to','for','with','and','or','but',
        'since','days','day','today','yesterday','morning','evening','afternoon',
        'night','hours','hour','time','mild','severe','acute','chronic','recent','old','new'
    }
    final = []
    for t in tokens:
        t = t.strip(STRIP_CHARS).lower()
        if t and len(t) > 1 and not t.isdigit() and t not in stop_words:
            final.append(t)

    seen = set()
    out = []
    for tok in final:
        if tok not in seen:
            out.append(tok)
            seen.add(tok)

    # Non-clinical noise - cleaned up duplicates and added part/body
    non_clinical_noise = {
        'outside', 'home', 'office', 'bus', 'travel', 'native', 'chores', 'household',
        'lunch', 'dinner', 'eating', 'due', 'wards', 'scin', 'dont', 'came', 'come', 'coming',
        'patient', 'pt', 'having', 'feeling', 'felt', 'is', 'was', 'has', 'have',
        'pm', 'am', 'hrs', 'hr', '1hrs', '2hrs', '12pm', 'back', 'one', 'two', 'three',
        'four', 'five', 'six', 'times', 'time', '3to', 'upper', 'regular',
        'cycle', 'lmp', 'lmppt', 'dat', '6th', '28th', 'episode', 'food', 'over', 'side',
        'ill', 'fitting', 'shoe', 'last', 'min', '1-2', 'both', 'leg', 'part', 'body'
    }
    out = [t for t in out if t not in non_clinical_noise]

    # Final safety: remove any remaining pure numeric tokens
    out = [t for t in out if not re.fullmatch(r'\d+', t)]

    text = " ".join(out)
    text = re.sub(r'\s+', ' ', text).strip()

    return text if text else "empty"

# quick sanity test
if __name__ == "__main__":
    samples = [
        'pt c/o fevr and hedache since 2 days, no cold, negative for malaria',
        '6th dat period high flow regular cycle 28th lmppt c/o fevr and hedache since 2 days, no cold, negative for malaria',
        'lass of sleep 3to 4 days',
        'heart bum for 1hrs',
        'episode loose stool due outside food',
        'Patient denies chest pain but complains of cough and fever',
        'no fever but abdominal pain since 2 days',
        'SHOE BITE ON BOTH LEG DUE TO ILL FEETING SHOE',
        'back and side part of the body'
    ]
    for s in samples:
        print("RAW:", s)
        print("PROC:", clean_text(s))
        print("---")