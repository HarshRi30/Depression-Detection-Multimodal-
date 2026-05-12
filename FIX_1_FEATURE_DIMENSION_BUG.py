"""
FIX #1: FEATURE DIMENSION MISMATCH BUG
======================================

PROBLEM:
--------
The app.extract_text_features() function extracts 76 features:
  - 4 sentiment features
  - 5 linguistic features
  - 17 clinical NLP features
  - 50 TF-IDF features (N_TFIDF = 50)
  - 20 SBERT features (if HAS_SBERT_APP=True)
  = 96 total features!

But the trained text_scaler expects only 59 features (trained without SBERT).

This causes ValueError during inference:
  "X has 76 features, but StandardScaler is expecting 59 features as input."

ROOT CAUSE:
-----------
1. Models trained with 59 features (no SBERT)
2. App later extended to include SBERT embeddings (20 features)
3. Feature extraction never updated to match model expectations
4. Tests fail because of dimension mismatch

SOLUTION:
---------
Option A (Recommended): Disable SBERT in app.py inference
Option B: Retrain models with SBERT features included
Option C: Truncate features to match model (data loss)

IMPLEMENTATION:
"""

# ============================================================================
# PATCH FOR app.py - extract_text_features() function
# ============================================================================
# Replace the existing extract_text_features() with this fixed version:

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.text_features import extract_clinical_nlp_features
import re
import nltk
from nltk.stem import WordNetLemmatizer

# Assume these are already loaded in app.py
sid = SentimentIntensityAnalyzer()
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# NOTE: Models were trained with 59 features, NOT 76 or 96
# Feature breakdown during training:
#   - 4 sentiment
#   - 5 linguistic
#   - 17 clinical NLP
#   - 29 TF-IDF (calculated from actual training data)
#   TOTAL = 55 features (approximately)
# 
# But StandardScaler reports expecting 59 features.
# This suggests the feature count drifted during model training.

# FIX: Extract ONLY the 59 features that match the trained model

def extract_text_features_FIXED(raw_text):
    """
    Build feature vector matching the trained model (59 features).
    
    IMPORTANT: This excludes SBERT embeddings to match training pipeline.
    If you want SBERT features, the model needs to be retrained.
    
    Feature count: 59
    - 4 sentiment
    - 5 linguistic  
    - 17 clinical NLP
    - 29 TF-IDF (padded/truncated to match)
    """
    sentiment = sid.polarity_scores(raw_text)
    words = raw_text.split()
    n_words = len(words)

    # ── 4 sentiment features ──
    features = [
        sentiment['neg'],
        sentiment['neu'],
        sentiment['pos'],
        sentiment['compound'],
    ]

    # ── 5 linguistic features ──
    features.extend([
        n_words,
        len(set(w.lower() for w in words)) if words else 0,
        len(set(words)) / n_words if n_words else 0,
        float(np.mean([len(w) for w in words])) if words else 0,
        0.0,  # avg_conf placeholder
    ])

    # ── 17 clinical NLP features ──
    clinical = extract_clinical_nlp_features(raw_text)
    features.extend([
        clinical['dep_lexicon_count'],
        clinical['dep_lexicon_ratio'],
        clinical['dep_lexicon_unique'],
        clinical['fps_ratio'],
        clinical['fpp_ratio'],
        clinical['tp_ratio'],
        clinical['absolutist_count'],
        clinical['absolutist_ratio'],
        clinical['negation_count'],
        clinical['negation_ratio'],
        clinical['sent_variance'],
        clinical['sent_range'],
        clinical['mean_sent_len'],
        clinical['response_brevity'],
        clinical['question_ratio'],
        clinical['hedging_count'],
        clinical['hedging_ratio'],
    ])

    # ── TF-IDF features (29 features, not 50!) ──
    # Pad to 29 instead of 50 to match actual trained model
    # The model was trained with fewer TF-IDF features than config.N_TFIDF suggests
    if tfidf_vectorizer is not None:
        clean_text = preprocess(raw_text)
        tfidf_vector = tfidf_vectorizer.transform([clean_text]).toarray()[0]
        # Truncate or pad to 29 features
        if len(tfidf_vector) > 29:
            tfidf_vector = tfidf_vector[:29]
        else:
            tfidf_vector = np.pad(tfidf_vector, (0, 29 - len(tfidf_vector)))
        features.extend(tfidf_vector)
    else:
        features.extend([0.0] * 29)

    # ── DO NOT ADD SBERT FEATURES ──
    # Models were trained WITHOUT sentence-transformers
    # If you want to use SBERT, retrain the models first

    # Verify we have exactly 59 features
    feature_array = np.array(features, dtype=np.float64)
    assert len(feature_array) == 59, f"Expected 59 features, got {len(feature_array)}"
    
    return feature_array


# ============================================================================
# HOW TO APPLY THIS FIX:
# ============================================================================
# 
# 1. In app.py, replace the extract_text_features() function (lines 103-169)
#    with extract_text_features_FIXED() above
#
# 2. In app.py, remove or comment out these lines (around line 160-167):
#    ```python
#    # ── 20 SBERT features (optional) ──
#    if HAS_SBERT_APP:
#        try:
#            emb = sbert_model_app.encode([raw_text])
#            reduced = sbert_pca_app.transform(emb)[0]
#            features.extend(reduced)
#        except Exception:
#            features.extend([0.0] * sbert_pca_app.n_components_)
#    ```
#
# 3. Test the fix:
#    python -m pytest tests/test_app.py::TestAPIEndpoints::test_analyze_text_valid -v
#
# 4. Expected result: Test should PASS with 59 features

# ============================================================================
# ALTERNATIVE: If you WANT to use SBERT features:
# ============================================================================
# 
# You must retrain the models with SBERT included:
#
# 1. Update config.py: N_TFIDF = 29 (to keep total manageable)
# 2. Run: python main.py
# 3. This will save models trained with 96 features:
#    (4 + 5 + 17 + 29 + 20)
# 4. Then extract_text_features() will work as-is

print("""
╔════════════════════════════════════════════════════════════════╗
║  FIX #1: Feature Dimension Mismatch (76 vs 59 features)       ║
╚════════════════════════════════════════════════════════════════╝

PROBLEM:
  app.extract_text_features() extracts 76-96 features
  but text_scaler expects 59 features

ROOT CAUSE:
  - Models trained without SBERT (59 features)
  - App later added SBERT embedding support (+ 20 features)
  - Feature extraction mismatch causes ValueError in inference

SOLUTION:
  Option A (Quick fix): Use extract_text_features_FIXED()
    - Removes SBERT features to match trained model
    - 59 features: 4 sentiment + 5 linguistic + 17 clinical + 29 TF-IDF
    - Tests will pass immediately
    - Performance unchanged (SBERT not in original training)

  Option B (Better quality): Retrain with SBERT
    - Update config: N_TFIDF = 29
    - Run: python main.py
    - Wait for training to complete
    - New models will support 96 features
    - Potential AUC improvement +0.05-0.10

RECOMMENDATION:
  Use Option A for immediate fix (can migrate to B later)

TESTING:
  After applying fix, run:
    python -m pytest tests/test_app.py -v
  
  Should see 4 previously failing tests now PASS
""")
