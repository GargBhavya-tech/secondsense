# Sherpa-onnx offline ASR — activation checklist (Phase 4 voice, non-QNN path)

The voice **grounding** half (spoken noun → COCO detection → steering) works on the default
build already. This folder adds the **ASR** half — turning *"chair"* spoken aloud into the
word `chair`, fully offline — via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
keyword spotting. It is OFF by default so `gradlew assembleDebug` stays green with no extra
files. `VoiceRecognizers.create()` falls back to the QNN Whisper stub until this is in place.

## 1. Kotlin wrapper sources

From the sherpa-onnx repo, copy the Android/Kotlin API wrapper `.kt` files into:

```
android/app/src/sherpa/kotlin/com/k2fsa/sherpa/onnx/
```

They live in the repo under `sherpa-onnx/kotlin-api/` (e.g. `KeywordSpotter.kt`,
`OnlineStream.kt`, `FeatureConfig.kt`, and the model-config files). Match the version to the
`.so` in step 2. Do **not** edit `SherpaKwsRecognizer.kt` — if it fails to compile, the config
field names in the wrapper drifted; adjust the call in `SherpaKwsRecognizer.initialize()`.

## 2. Native library

Download the prebuilt Android libs for the same version from the
[sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) (the
`sherpa-onnx-<version>-android.tar.bz2` / per-ABI `jniLibs` bundle) and place the `.so` files
(`libsherpa-onnx-jni.so` and its `libonnxruntime.so` etc.) under:

```
android/app/src/main/jniLibs/arm64-v8a/     <- the iQOO 15
android/app/src/main/jniLibs/x86_64/        <- emulator, optional
```

`src/main/jniLibs/` is already on the default jniLibs path, so no gradle change is needed for
the `.so`s.

## 3. KWS model

Download **`sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01`** (English, ~15 MB) from the
sherpa-onnx pretrained-models page
(<https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html>) and unpack its files
into:

```
android/app/src/main/assets/kws/
  encoder-...onnx        (matched by the substring "encoder")
  decoder-...onnx        (      "        "        "  "decoder")
  joiner-...onnx         (      "        "        "  "joiner")
  tokens.txt
  keywords.txt           <- YOU write this (see below)
```

If the model fails to load, add `androidResources { noCompress += "onnx" }` to
`app/build.gradle.kts` (left out by default to keep that block untouched).

### keywords.txt

One target per line, in the tokenized form sherpa expects (use
`sherpa-onnx-cli text2token` against `tokens.txt`, or start from the model's sample
`keywords.txt`). Keep the list to the COCO nouns `GoalGrounding` can resolve, e.g.:

```
chair
person
bottle
cup
laptop
backpack
book
```

`door` is not a COCO class — leave it out (or include it and accept that it spots but grounds
nothing; the HUD says "not in view / not a COCO class").

## 4. Build

```
gradlew :app:assembleDebug -PenableSherpa=true
```

On launch, logcat tag `SecondSense/sherpa` prints `ready (encoder=...)`. Tap **Find… (voice)**
in Scan/Seek mode, say a keyword, and the goal line in the HUD should switch to
`GOAL: chair → steering` when a chair is in frame.
