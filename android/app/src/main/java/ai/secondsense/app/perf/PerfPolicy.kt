package ai.secondsense.app.perf

/**
 * Thermal state, fused from Android's thermal status + headroom + battery temperature + the
 * inference latency we actually observe. Ordered coolest -> hottest.
 */
enum class ThermalTier { NOMINAL, WARM, HOT, CRITICAL }

/**
 * The performance knobs the app turns down as the SoC heats up inside the harness. Every
 * field is an integer "every Nth frame" cadence or a boolean load switch — no field ever
 * goes to 0, so the safety-critical detect+depth+drop-off path NEVER fully stops, it only
 * gets coarser.
 *
 * @param frameEveryN  process every Nth CameraX analysis frame (1 = every frame ~30fps,
 *                     2 ~= 15fps, 3 ~= 10fps, 6 ~= 5fps).
 * @param depthEveryN  run Depth-Anything every Nth processed frame (reuse the last map between).
 * @param hazardEveryN run the RANSAC/Hough drop-off fusion every Nth processed frame.
 * @param auxEnabled   ML Kit OCR + face ("read the sign", "person facing you") — first to shed.
 * @param yamnetEnabled the ambient hazard-sound listener (horn/siren) — shed second.
 * @param lowRes       request 320x240 camera analysis instead of 640x480 on the next bind.
 * @param label        short human string for the HUD / dashboard.
 */
data class PerfPolicy(
    val frameEveryN: Int,
    val depthEveryN: Int,
    val hazardEveryN: Int,
    val auxEnabled: Boolean,
    val yamnetEnabled: Boolean,
    val lowRes: Boolean,
    val label: String,
) {
    companion object {
        /**
         * The policy for a given [tier] and whether the user is [walking]. Idle always relaxes
         * one notch further than walking — a stopped user is not about to step off a curb — but
         * detection is kept alive either way.
         *
         * SAFETY FLOOR: at CRITICAL + walking, frameEveryN caps at 3 (~10 fps) and depthEveryN
         * at 6; the drop-off path stays responsive. What CRITICAL sheds is auxiliary load
         * (OCR/face, then YamNet) and resolution — never the core loop.
         */
        fun policyFor(tier: ThermalTier, walking: Boolean): PerfPolicy = when (tier) {
            ThermalTier.NOMINAL -> if (walking)
                PerfPolicy(1, 2, 2, auxEnabled = true, yamnetEnabled = true, lowRes = false, label = "nominal")
            else
                // Idle + cool: keep detection full-rate (a standing user shouldn't feel lag),
                // just halve the depth / drop-off fusion work.
                PerfPolicy(1, 3, 3, auxEnabled = true, yamnetEnabled = true, lowRes = false, label = "nominal-idle")

            ThermalTier.WARM -> if (walking)
                PerfPolicy(2, 3, 2, auxEnabled = true, yamnetEnabled = true, lowRes = false, label = "warm")
            else
                PerfPolicy(3, 4, 3, auxEnabled = true, yamnetEnabled = true, lowRes = false, label = "warm-idle")

            ThermalTier.HOT -> if (walking)
                PerfPolicy(2, 4, 3, auxEnabled = false, yamnetEnabled = true, lowRes = true, label = "hot")
            else
                PerfPolicy(4, 6, 4, auxEnabled = false, yamnetEnabled = true, lowRes = true, label = "hot-idle")

            ThermalTier.CRITICAL -> if (walking)
                PerfPolicy(3, 6, 4, auxEnabled = false, yamnetEnabled = false, lowRes = true, label = "critical")
            else
                PerfPolicy(6, 8, 6, auxEnabled = false, yamnetEnabled = false, lowRes = true, label = "critical-idle")
        }
    }
}
