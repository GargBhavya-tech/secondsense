package ai.secondsense.app.voice

import java.security.MessageDigest

/**
 * Tier-2 safety-intent gate — a small LEARNED classifier (hashed char 3+4-grams -> logistic
 * regression, weights in [SafetyGateWeights], trained by laptop/tools/train_safety_gate.py on
 * EN/HI/KN safety questions vs everything else). ~37 KB, no model file, no tokenizer, works in
 * any script because the features are character n-grams.
 *
 * Runs BEFORE [IntentInterpreter] in MainActivity: if [isSafetyQuery] fires, the spoken line
 * skips the grammar AND the LLM and gets the deterministic [MainActivity.speakSafety]
 * deflection. Tuned for recall (threshold 0.50) — a false positive just means an extra
 * deflection, which is the fail-safe outcome.
 */
class NgramSafetyGate : SafetyVectorGate {

    private val md = MessageDigest.getInstance("MD5")

    override fun isReady(): Boolean = true

    override fun isSafetyQuery(transcript: String): Boolean {
        if (transcript.isBlank()) return false
        val v = featurize(transcript)
        var z = SafetyGateWeights.BIAS
        val w = SafetyGateWeights.W
        for (i in v.indices) if (v[i] != 0f) z += v[i] * w[i]
        val p = 1f / (1f + Math.exp(-z.toDouble()).toFloat())
        return p >= SafetyGateWeights.THRESHOLD
    }

    /** MUST match featurize() in train_safety_gate.py exactly. */
    private fun featurize(text: String): FloatArray {
        val t = " " + text.lowercase().trim() + " "
        val v = FloatArray(SafetyGateWeights.DIM)
        for (n in intArrayOf(3, 4)) {
            var i = 0
            while (i + n <= t.length) {
                v[bucket(t.substring(i, i + n))] = 1f
                i++
            }
        }
        var ss = 0.0
        for (x in v) ss += (x * x).toDouble()
        val nrm = Math.sqrt(ss).toFloat()
        if (nrm > 0f) for (i in v.indices) v[i] /= nrm
        return v
    }

    private fun bucket(gram: String): Int {
        val d = md.digest(gram.toByteArray(Charsets.UTF_8))   // 16 bytes
        md.reset()
        // first 4 bytes as a big-endian unsigned int (== Python int(hexdigest()[:8], 16))
        val u = ((d[0].toLong() and 0xFF) shl 24) or
            ((d[1].toLong() and 0xFF) shl 16) or
            ((d[2].toLong() and 0xFF) shl 8) or
            (d[3].toLong() and 0xFF)
        return (u % SafetyGateWeights.DIM).toInt()
    }
}
