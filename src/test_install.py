# print("Script is starting...")
import pandas as pd
import spacy
from symspellpy.symspellpy import SymSpell
from rapidfuzz import process

print("✅ pandas version:", pd.__version__)
print("✅ spaCy version:", spacy.__version__)

nlp = spacy.load("en_core_web_sm")
doc = nlp("Patient denies fever but has headache")

print("✅ spaCy tokenization:", [token.text for token in doc])

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
print("✅ SymSpell initialized")

print("✅ RapidFuzz similarity:", process.extractOne("fevr", ["fever", "headache"]))
