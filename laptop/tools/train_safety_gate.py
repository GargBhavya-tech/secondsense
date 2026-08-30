"""
Trains the Tier-2 safety-intent classifier and bakes the weights into a Kotlin file.

Model: hashed character 3+4-grams (span any script -> multilingual) -> L2-normalised sparse
vector -> logistic regression (numpy, hand-rolled, L2-reg full-batch GD). ~24 KB of weights,
no runtime deps on Android.

    python laptop/tools/train_safety_gate.py

Writes: android/app/src/main/java/ai/secondsense/app/voice/SafetyGateWeights.kt
Prints held-out accuracy + the false-negative / false-positive lists (must be ~0 FN).
"""
from __future__ import annotations

import hashlib
import pathlib
import random

import numpy as np

DIM = 4096
RNG = random.Random(42)

# --------------------------------------------------------------------------- data
SAFETY = [
    "is it safe to cross", "is it safe to cross the road", "is it safe to cross the street",
    "is it safe to walk", "is it safe to walk now", "is it safe to go", "is it safe to move",
    "is it safe to move forward", "is it safe now", "is it safe to proceed",
    "can i cross", "can i cross now", "can i cross the road", "can i cross here",
    "can i walk", "can i walk now", "can i go", "can i go now", "can i move", "can i proceed",
    "can i start walking", "may i cross", "may i proceed", "should i cross", "should i go",
    "should i walk", "should i move now", "is it okay to cross", "is it okay to walk",
    "is it ok to go", "am i clear to go", "am i safe to move", "am i good to walk",
    "is now a good time to cross", "tell me when it is safe to cross",
    "is it clear to cross", "is it clear to go", "is it clear to move",
    "is the path clear", "is the way clear", "is the way ahead clear", "is the road clear",
    "is the coast clear", "is the crosswalk clear", "is the pavement clear",
    "is anything in my way", "is there anything in my way", "is anything blocking my path",
    "is something in front of me", "are there obstacles ahead", "any obstacles ahead",
    "is there a step ahead", "is there a drop ahead", "is there a curb ahead",
    "will i bump into anything", "will i hit anything if i walk", "will i trip on anything",
    "is it clear in front of me", "anything i should worry about ahead", "is it blocked ahead",
    "are there cars coming", "are there cars ahead", "is a car coming",
    "how is the traffic", "is the traffic clear", "is it clear ahead",
    "क्या सड़क पार करना सुरक्षित है", "क्या अभी सड़क पार करना सुरक्षित है",
    "क्या मैं अभी सड़क पार कर सकता हूँ", "क्या मैं अभी चल सकता हूँ",
    "क्या मैं आगे बढ़ सकता हूँ", "क्या आगे बढ़ना ठीक है", "क्या आगे बढ़ना सुरक्षित है",
    "क्या रास्ता साफ है", "क्या रास्ता खाली है", "क्या आगे का रास्ता साफ है",
    "क्या आगे कोई रुकावट है", "क्या सामने कुछ है", "क्या सामने कोई रुकावट है",
    "क्या मैं जा सकता हूँ", "क्या अभी चलना ठीक है", "क्या अभी चलना सुरक्षित है",
    "आगे कोई सीढ़ी या गड्ढा तो नहीं", "क्या कोई गाड़ी आ रही है", "क्या रास्ते में कुछ है",
    "kya sadak paar karna safe hai", "kya sadak paar karna theek hai",
    "kya main ab chal sakta hoon", "kya main aage badh sakta hoon", "kya raasta saaf hai",
    "kya raasta khaali hai", "aage kuch hai kya", "kya aage badhna theek hai",
    "kya koi gaadi aa rahi hai", "kya main ab jaa sakta hoon",
    "ರಸ್ತೆ ದಾಟುವುದು ಸುರಕ್ಷಿತವೇ", "ಈಗ ರಸ್ತೆ ದಾಟುವುದು ಸುರಕ್ಷಿತವೇ", "ನಾನು ಈಗ ನಡೆಯಬಹುದೇ",
    "ನಾನು ಮುಂದೆ ಹೋಗಬಹುದೇ", "ದಾರಿ ಸ್ಪಷ್ಟವಾಗಿದೆಯೇ", "ದಾರಿ ಖಾಲಿಯಾಗಿದೆಯೇ",
    "ಮುಂದೆ ಏನಾದರೂ ಅಡ್ಡಿ ಇದೆಯೇ", "ಮುಂದೆ ಏನಾದರೂ ಇದೆಯೇ", "ಈಗ ದಾಟುವುದು ಸರಿಯೇ",
    "ಕಾರು ಬರುತ್ತಿದೆಯೇ",
]

NOT_SAFETY = [
    "find my keys", "find the exit", "find the nearest chair", "take me to the door",
    "guide me to the stairs", "look for a bottle", "locate the bench", "where is the elevator",
    "where did i leave my wallet", "where did i put my phone", "where is my bag",
    "read this", "read the sign", "what does it say", "read that label", "read out the menu",
    "whats ahead", "what is around me", "describe the scene", "what do you see",
    "whats in front of me", "look around", "tell me whats there",
    "status", "how am i doing", "where am i", "whats my battery", "what mode am i in",
    "give me a status report", "system status",
    "repeat", "say that again", "what did you say", "come again",
    "stop the beeping", "mute", "be quiet", "turn off the sound", "turn on cues",
    "start the cues", "quieter please", "louder please",
    "pause", "hold on", "resume", "carry on", "continue please",
    "never mind", "cancel", "stop looking", "forget it",
    "im sitting down", "getting on the bus", "lets start walking", "im at home now",
    "ive stopped", "conversation mode", "im on the train",
    "speak hindi", "switch to english", "talk in hindi",
    "call mom", "phone dad", "call my wife", "set a timer for five minutes",
    "timer for thirty seconds", "set a ten minute timer",
    "help", "what can i say", "what can you do",
    "the weather today is quite nice", "i really like coffee in the morning",
    "what time is it right now", "tell me a joke", "how are you doing today",
    "play some music", "whats the news", "remind me to buy milk",
    "my favourite colour is blue", "the movie was great last night",
    "मेरा फ़ोन कहाँ है", "मेरी चाबियाँ ढूंढो", "साइन पढ़ो", "आवाज़ बंद करो",
    "माँ को फ़ोन करो", "आज मौसम अच्छा है", "मुझे कॉफ़ी पसंद है",
    "ನನ್ನ ಕೀಲಿಗಳನ್ನು ಹುಡುಕಿ", "ಸೈನ್ ಓದಿ", "ಸಂಗೀತ ಹಾಕಿ", "ಇಂದು ಹವಾಮಾನ ಚೆನ್ನಾಗಿದೆ",
]

PRE = ["", "", "hey ", "please ", "um ", "so ", "ok "]
POST = ["", "", " please", " now", " right now", " here"]


def augment(phrases):
    out = set()
    for p in phrases:
        out.add(p)
        for _ in range(3):
            out.add((RNG.choice(PRE) + p + RNG.choice(POST)).strip())
    return sorted(out)


def featurize(text: str) -> np.ndarray:
    t = " " + text.lower().strip() + " "
    v = np.zeros(DIM, np.float32)
    for n in (3, 4):
        for i in range(len(t) - n + 1):
            g = t[i:i + n]
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16) % DIM
            v[h] = 1.0
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def train():
    pos = augment(SAFETY)
    neg = augment(NOT_SAFETY)
    X = np.stack([featurize(p) for p in pos + neg])
    y = np.array([1] * len(pos) + [0] * len(neg), np.float32)
    names = np.array(pos + neg, dtype=object)
    order = np.random.default_rng(0).permutation(len(y))
    X, y, names = X[order], y[order], names[order]
    cut = int(len(y) * 0.82)
    Xtr, ytr, Xte, yte, nte = X[:cut], y[:cut], X[cut:], y[cut:], names[cut:]

    w = np.zeros(DIM, np.float32); b = 0.0
    lr, l2 = 0.5, 1e-4
    for _ in range(4000):
        p = 1.0 / (1.0 + np.exp(-(Xtr @ w + b)))
        g = p - ytr
        w -= lr * (Xtr.T @ g / len(ytr) + l2 * w)
        b -= lr * g.mean()

    # Bias toward recall: a false "this is a safety question" just triggers the deterministic
    # deflection (fail-safe, per the research). Cap the threshold low.
    ptr = 1.0 / (1.0 + np.exp(-(Xtr @ w + b)))
    thr = min(0.50, max(0.32, float(np.percentile(ptr[ytr == 1], 3)) - 0.05))

    pte = 1.0 / (1.0 + np.exp(-(Xte @ w + b)))
    pred = (pte >= thr).astype(int)
    fn = [nte[i] for i in range(len(yte)) if yte[i] == 1 and pred[i] == 0]
    fp = [nte[i] for i in range(len(yte)) if yte[i] == 0 and pred[i] == 1]
    print(f"held-out acc {float((pred == yte).mean()):.3f}  thr {thr:.3f}  "
          f"(pos {int(yte.sum())}, neg {int((1 - yte).sum())})")
    print(f"false negatives ({len(fn)}): {list(fn)}")
    print(f"false positives ({len(fp)}): {list(fp)}")
    return w, b, thr


def emit_kotlin(w, b, thr):
    dst = pathlib.Path(__file__).resolve().parents[2] / \
        "android/app/src/main/java/ai/secondsense/app/voice/SafetyGateWeights.kt"
    body = ",".join(f"{x:.5f}f" for x in np.round(w, 5))
    txt = f'''package ai.secondsense.app.voice

/**
 * GENERATED by laptop/tools/train_safety_gate.py — do not edit by hand.
 * Tier-2 safety-intent classifier: hashed char 3+4-grams (DIM buckets) -> logistic regression.
 * Multilingual via char n-grams (English / Hindi / Kannada in the training set).
 */
internal object SafetyGateWeights {{
    const val DIM = {DIM}
    const val BIAS = {b:.6f}f
    const val THRESHOLD = {thr:.4f}f
    val W: FloatArray = floatArrayOf(
        {body}
    )
}}
'''
    dst.write_text(txt, encoding="utf-8")
    print("wrote", dst, f"({dst.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    w, b, thr = train()
    emit_kotlin(w, b, thr)
