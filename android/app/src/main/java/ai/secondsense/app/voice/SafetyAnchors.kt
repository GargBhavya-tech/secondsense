package ai.secondsense.app.voice

/**
 * Canonical "is it safe / can I move / is the way clear" phrasings, in English, Hindi and
 * Kannada, for the Tier-2 [SafetyVectorGate]. At init the embedding model turns each of these
 * into a vector; at query time the user's transcript is embedded and compared (max cosine
 * similarity) against the whole set. A hit above the tuned threshold classifies the query as
 * SAFETY_CRITICAL and bypasses the generative LLM.
 *
 * Curation rules (from the deep-research test plan):
 *  - Cover the INTENT ("may I proceed?"), not just the words. Vary verbs (cross/walk/go/move),
 *    subjects (I / the path / the road), politeness, and directness.
 *  - Include the ambiguous-but-should-still-deflect cases ("describe the crosswalk",
 *    "are there cars ahead") — a false positive here is a fail-SAFE degradation.
 *  - Do NOT include benign scene questions ("what colour is the door", "read that sign").
 *  - Keep it 40-200 lines; more anchors = smoother decision boundary. Extend freely.
 */
object SafetyAnchors {
    val PHRASES: List<String> = listOf(
        // --- English: direct safety verdicts ---
        "is it safe to cross",
        "is it safe to cross the road",
        "is it safe to walk now",
        "is it safe to go",
        "is it safe to move forward",
        "can I cross now",
        "can I cross the street",
        "can I walk forward",
        "can I go now",
        "can I start walking",
        "should I cross",
        "should I go now",
        "may I proceed",
        "is it okay to walk",
        "is it okay to cross here",
        "am I clear to go",
        "am I safe to move",
        "is now a good time to cross",
        "tell me when it is safe to cross",
        // --- English: path / obstacle framing ---
        "is the path clear",
        "is the way ahead clear",
        "is the road clear",
        "is the coast clear",
        "is there anything in my way",
        "is anything blocking my path",
        "will I bump into anything if I walk",
        "are there obstacles ahead",
        "is there a step or a drop ahead",
        "is it clear in front of me",
        "anything I should worry about ahead",
        "is the crosswalk clear",
        "are there cars coming",
        "are there cars parked ahead",
        "describe the crosswalk",
        "how is the traffic right now",
        // --- Hindi (Devanagari) ---
        "क्या सड़क पार करना सुरक्षित है",
        "क्या मैं अभी सड़क पार कर सकता हूँ",
        "क्या मैं अभी चल सकता हूँ",
        "क्या आगे बढ़ना ठीक है",
        "क्या रास्ता साफ है",
        "क्या आगे कोई रुकावट है",
        "क्या सामने कुछ है",
        "क्या मैं जा सकता हूँ",
        "क्या अभी चलना सुरक्षित है",
        "आगे कोई सीढ़ी या गड्ढा तो नहीं",
        "क्या कोई गाड़ी आ रही है",
        // --- Hindi (romanised, common in speech-to-text) ---
        "kya sadak paar karna safe hai",
        "kya main ab chal sakta hoon",
        "kya raasta saaf hai",
        "aage kuch hai kya",
        "kya aage badhna theek hai",
        // --- Kannada ---
        "ರಸ್ತೆ ದಾಟುವುದು ಸುರಕ್ಷಿತವೇ",
        "ನಾನು ಈಗ ನಡೆಯಬಹುದೇ",
        "ದಾರಿ ಸ್ಪಷ್ಟವಾಗಿದೆಯೇ",
        "ಮುಂದೆ ಏನಾದರೂ ಅಡ್ಡಿ ಇದೆಯೇ",
        "ಈಗ ದಾಟುವುದು ಸರಿಯೇ",
    )
}
