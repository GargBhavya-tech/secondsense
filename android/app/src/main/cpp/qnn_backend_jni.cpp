// QNN native bridge — JNI implementation behind ai.secondsense.app.inference.qnn.QnnBackend.
//
// STATUS: builds clean against the real QAIRT 2.49.0 headers (C:\Master_Brain\Projects\qairt).
// nativeInit/nativeLoadModel/nativeRun are fully implemented, including graphExecute tensor
// binding and real graph-name/shape discovery via QnnSystemContext (a compiled context
// binary's internal graph name has no relation to our "yolo"/"depth" model keys, so we read it
// out of the binary itself rather than guessing). NOT YET RUN on-device — next step is
// wiring NativeQnnBackend into EngineConfig and testing nativeInit()/load() against the real
// yolov11_det.bin/depth_anything_v2.bin assets on the iQOO 15.
//
// DESIGN: dlopen() the backend .so (libQnnHtp.so for the Hexagon HTP) at runtime rather than
// link against it at build time, because Qualcomm ships interchangeable backend .so files
// (HTP/CPU/GPU/DSP) and only whichever one is actually bundled in jniLibs/ needs to resolve.
// Everything downstream in the Kotlin decode layer is backend-agnostic; this file's only job
// is: load a context binary, run it, hand back flat float tensors.

#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Confirmed against the real QAIRT 2.49 headers (compiled clean, zero corrections needed) —
// see System/QnnSystemContext.h for the binary-introspection API used to discover real graph
// names and tensor shapes below, instead of guessing them.
#include "QnnInterface.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpDeviceConfigShared.h"
#include "System/QnnSystemContext.h"

#define LOG_TAG "SecondSense/qnn_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// ---- Global QNN handles (one backend, one device, shared across models — a real device only
// has one Hexagon HTP, so this mirrors StubQnnBackend's single-instance assumption). ----
void* g_backendLib = nullptr;
const QNN_INTERFACE_VER_TYPE* g_iface = nullptr;   // VERIFY AGAINST SDK: exact typedef name
Qnn_BackendHandle_t g_backendHandle = nullptr;
Qnn_DeviceHandle_t g_deviceHandle = nullptr;
bool g_ready = false;

// One tensor's shape + name, deep-copied out of the (transient) QnnSystemContext binary-info
// struct so it survives past QnnSystemContext_free — that's the only reason this exists
// instead of reusing Qnn_Tensor_t directly (its dimensions pointer would dangle).
struct TensorSpec {
    std::string name;
    Qnn_DataType_t dataType = QNN_DATATYPE_FLOAT_32;
    std::vector<uint32_t> dims;
    uint32_t elementCount() const {
        uint32_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
};

struct LoadedModel {
    Qnn_ContextHandle_t context = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    // Real shapes/names read out of the context binary itself (via QnnSystemContext) at load
    // time — NOT guessed from the model name, since the compiled graph's tensor names/shapes
    // are whatever qai_hub_models' export baked in, independent of our "yolo"/"depth" keys.
    std::vector<TensorSpec> inputs;
    std::vector<TensorSpec> outputs;
};

std::map<std::string, LoadedModel> g_models;

// Symbol exported by every QNN backend .so — this part IS stable across SDK versions, it's
// the documented entry point every backend implements.
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn)(
    const QnnInterface_t*** providerList, uint32_t* numProviders);

bool initInterface(const char* backendSoPath) {
    g_backendLib = dlopen(backendSoPath, RTLD_NOW | RTLD_LOCAL);
    if (!g_backendLib) {
        LOGE("dlopen(%s) failed: %s", backendSoPath, dlerror());
        return false;
    }
    auto getProviders = reinterpret_cast<QnnInterfaceGetProvidersFn>(
        dlsym(g_backendLib, "QnnInterface_getProviders"));
    if (!getProviders) {
        LOGE("QnnInterface_getProviders not found in %s", backendSoPath);
        return false;
    }
    const QnnInterface_t** providers = nullptr;
    uint32_t numProviders = 0;
    if (getProviders(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0) {
        LOGE("QnnInterface_getProviders returned no providers");
        return false;
    }
    // Confirmed against the real header: providers[0]->QNN_INTERFACE_VER_NAME is the
    // version-tagged union member holding the function table. First real provider is fine for
    // our single-backend use case.
    g_iface = &providers[0]->QNN_INTERFACE_VER_NAME;
    return true;
}

// libQnnSystem.so's introspection API (QnnSystemContext_*) is a set of PLAIN exported C
// functions (marked QNN_SYSTEM_API in the header) — no provider-interface indirection needed,
// unlike the backend .so. Used once per model load to discover the real compiled graph name
// and tensor shapes from the context binary itself, since a qai_hub_models export's internal
// graph name (e.g. "yolo26_det_float_yolo26s") has no relation to our own "yolo"/"depth" keys.
typedef Qnn_ErrorHandle_t (*QnnSystemContextCreateFn)(QnnSystemContext_Handle_t*);
typedef Qnn_ErrorHandle_t (*QnnSystemContextGetBinaryInfoFn)(
    QnnSystemContext_Handle_t, void*, uint64_t,
    const QnnSystemContext_BinaryInfo_t**, Qnn_ContextBinarySize_t*);
typedef Qnn_ErrorHandle_t (*QnnSystemContextFreeFn)(QnnSystemContext_Handle_t);

void* g_systemLib = nullptr;
QnnSystemContextCreateFn g_sysCreate = nullptr;
QnnSystemContextGetBinaryInfoFn g_sysGetBinaryInfo = nullptr;
QnnSystemContextFreeFn g_sysFree = nullptr;

bool ensureSystemLib() {
    if (g_systemLib) return true;
    // Bundled alongside the backend .so in jniLibs/arm64-v8a — see build.gradle.kts's
    // jniLibs wiring and the copy step documented in secondsense_phase7... (jniLibs README).
    g_systemLib = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    if (!g_systemLib) {
        LOGE("dlopen(libQnnSystem.so) failed: %s", dlerror());
        return false;
    }
    g_sysCreate = reinterpret_cast<QnnSystemContextCreateFn>(
        dlsym(g_systemLib, "QnnSystemContext_create"));
    g_sysGetBinaryInfo = reinterpret_cast<QnnSystemContextGetBinaryInfoFn>(
        dlsym(g_systemLib, "QnnSystemContext_getBinaryInfo"));
    g_sysFree = reinterpret_cast<QnnSystemContextFreeFn>(
        dlsym(g_systemLib, "QnnSystemContext_free"));
    if (!g_sysCreate || !g_sysGetBinaryInfo || !g_sysFree) {
        LOGE("libQnnSystem.so missing expected symbols");
        return false;
    }
    return true;
}

std::vector<uint32_t> dimsOf(const Qnn_Tensor_t& t) {
    // Both TensorV1 and TensorV2 have rank/dimensions at the same relative position, but the
    // union means we must branch on version to read the right member — VERIFY AGAINST SDK
    // confirmed both v1/v2 field names above; qai_hub_models context binaries observed using
    // V1 tensors for a plain float export.
    uint32_t rank = (t.version == QNN_TENSOR_VERSION_2) ? t.v2.rank : t.v1.rank;
    uint32_t* dims = (t.version == QNN_TENSOR_VERSION_2) ? t.v2.dimensions : t.v1.dimensions;
    std::vector<uint32_t> out(rank);
    for (uint32_t i = 0; i < rank; i++) out[i] = dims ? dims[i] : 0;
    return out;
}

const char* nameOf(const Qnn_Tensor_t& t) {
    return (t.version == QNN_TENSOR_VERSION_2) ? t.v2.name : t.v1.name;
}

Qnn_DataType_t dataTypeOf(const Qnn_Tensor_t& t) {
    return (t.version == QNN_TENSOR_VERSION_2) ? t.v2.dataType : t.v1.dataType;
}

// Byte size of one element for the data types our float-precision exports actually use.
// Anything unexpected falls back to 4 bytes (float32) rather than crashing — logged loudly so
// a real quantized model would be caught immediately instead of silently misreading tensors.
size_t elementBytes(Qnn_DataType_t dt) {
    switch (dt) {
        case QNN_DATATYPE_FLOAT_16: return 2;
        case QNN_DATATYPE_FLOAT_32: return 4;
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_INT_8: return 1;
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_INT_16: return 2;
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_INT_32: return 4;
        default:
            LOGE("elementBytes: unexpected dataType 0x%x, assuming 4 bytes", dt);
            return 4;
    }
}

} // namespace

extern "C" {

JNIEXPORT jboolean JNICALL
Java_ai_secondsense_app_inference_qnn_NativeQnnBackend_nativeInit(
    JNIEnv* env, jobject /*thiz*/, jstring backendSoPath) {
    const char* path = env->GetStringUTFChars(backendSoPath, nullptr);
    std::string pathStr(path);

    // REAL BUG FOUND on first device test: deviceCreate() failed with no path issue visible —
    // the HTP backend needs to find its DSP-side "skel" library (libQnnHtpV81Skel.so, the part
    // that actually runs ON the Hexagon core, distinct from the stub .so we dlopen here) via
    // Qualcomm's FastRPC loader, which searches ADSP_LIBRARY_PATH, not our app's normal linker
    // search path. Set it to the same directory backendSoPath lives in (our jniLibs extraction
    // dir), where the skel .so's were also bundled.
    size_t lastSlash = pathStr.find_last_of('/');
    std::string libDir = (lastSlash != std::string::npos) ? pathStr.substr(0, lastSlash) : ".";
    setenv("ADSP_LIBRARY_PATH", libDir.c_str(), 1);
    LOGI("ADSP_LIBRARY_PATH set to %s", libDir.c_str());

    bool ok = initInterface(path);
    env->ReleaseStringUTFChars(backendSoPath, path);
    if (!ok) return JNI_FALSE;

    // VERIFY AGAINST SDK: backendCreate signature — some SDK versions take a config array,
    // pass nullptr/empty for "use backend defaults" which is valid per the docs.
    Qnn_ErrorHandle_t backendErr = g_iface->backendCreate(nullptr, nullptr, &g_backendHandle);
    if (backendErr != QNN_SUCCESS) {
        LOGE("backendCreate failed, err=0x%llx (%lld)",
             (unsigned long long) backendErr, (long long) backendErr);
        return JNI_FALSE;
    }
    LOGI("backendCreate OK, handle=%p", (void*) g_backendHandle);
    // REAL BUG FOUND (2nd device-create failure, after ADSP_LIBRARY_PATH fixed the 1st): our
    // app isn't signed with Qualcomm's special key, so the HTP backend defaults to expecting a
    // "signed PD" (process domain) on the DSP and refuses unsigned code. Explicitly request
    // UNSIGNED PD — the documented mechanism for third-party/dev apps without that signature
    // (confirmed via the TFLite HTP delegate's equivalent `dsp_pd_session=unsigned` option in
    // options.html; this is the raw-QNN-API equivalent config).
    QnnHtpDevice_CustomConfig_t htpCustom = {};
    htpCustom.option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
    htpCustom.useSignedProcessDomain.deviceId = 0;
    htpCustom.useSignedProcessDomain.useSignedProcessDomain = false;

    QnnDevice_Config_t deviceConfig = {};
    deviceConfig.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    deviceConfig.customConfig = &htpCustom;

    const QnnDevice_Config_t* configArray[] = {&deviceConfig, nullptr};

    Qnn_ErrorHandle_t devErr = g_iface->deviceCreate(nullptr, configArray, &g_deviceHandle);
    if (devErr == 14001) {
        // INVALID_CONFIG with our custom config — retry with NO config at all (nullptr), to
        // isolate whether the base device itself is reachable independent of the PD setting.
        LOGI("deviceCreate(unsigned-PD config) -> INVALID_CONFIG; retrying with config=nullptr "
             "to isolate the cause");
        devErr = g_iface->deviceCreate(nullptr, nullptr, &g_deviceHandle);
    }
    if (devErr != QNN_SUCCESS) {
        LOGE("deviceCreate failed, err=0x%llx (%lld) even with unsigned-PD config — check "
             "ADSP_LIBRARY_PATH (%s) actually contains a matching libQnnHtpV<N>Skel.so for "
             "this device's real Hexagon version",
             (unsigned long long) devErr, (long long) devErr, libDir.c_str());
        return JNI_FALSE;
    }
    g_ready = true;
    LOGI("QNN backend + device created OK");
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_ai_secondsense_app_inference_qnn_NativeQnnBackend_nativeLoadModel(
    JNIEnv* env, jobject /*thiz*/, jstring modelName, jobject byteBuffer) {
    if (!g_ready) {
        LOGE("nativeLoadModel called before nativeInit succeeded");
        return JNI_FALSE;
    }
    const char* name = env->GetStringUTFChars(modelName, nullptr);
    std::string modelKey(name);
    env->ReleaseStringUTFChars(modelName, name);

    auto* binaryData = static_cast<uint8_t*>(env->GetDirectBufferAddress(byteBuffer));
    jlong binarySize = env->GetDirectBufferCapacity(byteBuffer);
    if (!binaryData || binarySize <= 0) {
        LOGE("nativeLoadModel(%s): input ByteBuffer must be direct (see Kotlin caller)",
             modelKey.c_str());
        return JNI_FALSE;
    }

    if (!ensureSystemLib()) return JNI_FALSE;

    // --- Discover the REAL compiled graph name + tensor shapes from the binary itself first.
    // A qai_hub_models export's internal graph name (observed: "yolo26_det_float_yolo26s") has
    // no relation to our own "yolo"/"depth" keys — guessing it would silently fail
    // graphRetrieve, so we read it out instead. ---
    LoadedModel model;
    {
        QnnSystemContext_Handle_t sysCtx = nullptr;
        if (g_sysCreate(&sysCtx) != QNN_SUCCESS) {
            LOGE("QnnSystemContext_create failed for %s", modelKey.c_str());
            return JNI_FALSE;
        }
        const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
        Qnn_ContextBinarySize_t infoSize = 0;
        Qnn_ErrorHandle_t err = g_sysGetBinaryInfo(
            sysCtx, binaryData, static_cast<uint64_t>(binarySize), &binaryInfo, &infoSize);
        if (err != QNN_SUCCESS || !binaryInfo) {
            LOGE("QnnSystemContext_getBinaryInfo failed for %s (err=%lld)",
                 modelKey.c_str(), (long long) err);
            g_sysFree(sysCtx);
            return JNI_FALSE;
        }
        if (binaryInfo->version != QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
            // V2/V3 add updatable-tensor / graph-blob fields we don't need for a plain float
            // export; V1's graphs[]/graphInfoV1 layout is a prefix-compatible subset in
            // practice for our purposes, but log loudly since this is the one place a newer
            // QAIRT SDK version could genuinely change behavior.
            LOGI("Binary info version %d != V1 (expected for a plain float export) — "
                 "reading via V1 layout anyway, verify if this model behaves unexpectedly",
                 (int) binaryInfo->version);
        }
        const auto& v1 = binaryInfo->contextBinaryInfoV1;
        if (v1.numGraphs == 0 || !v1.graphs) {
            LOGE("Context binary for %s has no graphs", modelKey.c_str());
            g_sysFree(sysCtx);
            return JNI_FALSE;
        }
        // qai_hub_models exports compile exactly one graph per context binary — take the first.
        const auto& g = v1.graphs[0].graphInfoV1;
        std::string realGraphName = g.graphName ? g.graphName : "";
        LOGI("Model '%s': real compiled graph name = '%s' (%u inputs, %u outputs)",
             modelKey.c_str(), realGraphName.c_str(), g.numGraphInputs, g.numGraphOutputs);

        for (uint32_t i = 0; i < g.numGraphInputs; i++) {
            TensorSpec spec;
            spec.name = nameOf(g.graphInputs[i]);
            spec.dataType = dataTypeOf(g.graphInputs[i]);
            spec.dims = dimsOf(g.graphInputs[i]);
            model.inputs.push_back(spec);
        }
        for (uint32_t i = 0; i < g.numGraphOutputs; i++) {
            TensorSpec spec;
            spec.name = nameOf(g.graphOutputs[i]);
            spec.dataType = dataTypeOf(g.graphOutputs[i]);
            spec.dims = dimsOf(g.graphOutputs[i]);
            model.outputs.push_back(spec);
        }
        g_sysFree(sysCtx); // frees binaryInfo too — we've already deep-copied what we need.

        // --- Now actually load the binary into the backend/device and retrieve the graph. ---
        if (g_iface->contextCreateFromBinary(
                g_backendHandle, g_deviceHandle, nullptr,
                binaryData, static_cast<uint64_t>(binarySize),
                &model.context, nullptr) != QNN_SUCCESS) {
            LOGE("contextCreateFromBinary failed for %s", modelKey.c_str());
            return JNI_FALSE;
        }
        if (g_iface->graphRetrieve(model.context, realGraphName.c_str(), &model.graph) !=
            QNN_SUCCESS) {
            LOGE("graphRetrieve('%s') failed for model '%s'",
                 realGraphName.c_str(), modelKey.c_str());
            return JNI_FALSE;
        }
    }

    g_models[modelKey] = std::move(model);
    LOGI("Loaded QNN model '%s' (%lld bytes)", modelKey.c_str(), (long long) binarySize);
    return JNI_TRUE;
}

JNIEXPORT jobjectArray JNICALL
Java_ai_secondsense_app_inference_qnn_NativeQnnBackend_nativeRun(
    JNIEnv* env, jobject /*thiz*/, jstring modelName, jobject inputBuffer) {
    const char* name = env->GetStringUTFChars(modelName, nullptr);
    std::string modelKey(name);
    env->ReleaseStringUTFChars(modelName, name);

    auto it = g_models.find(modelKey);
    if (it == g_models.end()) {
        LOGE("nativeRun: model '%s' not loaded", modelKey.c_str());
        return env->NewObjectArray(0, env->FindClass("java/lang/Object"), nullptr);
    }
    LoadedModel& model = it->second;
    jclass objClass = env->FindClass("java/lang/Object");

    auto* inputData = static_cast<uint8_t*>(env->GetDirectBufferAddress(inputBuffer));
    if (!inputData) {
        LOGE("nativeRun(%s): input ByteBuffer must be direct", modelKey.c_str());
        return env->NewObjectArray(0, objClass, nullptr);
    }
    if (model.inputs.size() != 1) {
        // Every model in this pipeline (YOLO, depth) has exactly one image input — a second
        // input would need per-tensor offsets into inputBuffer, deliberately unsupported here
        // rather than guessed at.
        LOGE("nativeRun(%s): expected exactly 1 input tensor, graph has %zu",
             modelKey.c_str(), model.inputs.size());
        return env->NewObjectArray(0, objClass, nullptr);
    }

    // --- Build the input Qnn_Tensor_t, pointing straight at the caller's direct buffer (no
    // copy) --- using TensorV1 (QNN_TENSOR_VERSION_1), matching what the binary's own tensors
    // were read as via QnnSystemContext (see dimsOf/nameOf — V1 fields at load time).
    std::vector<uint32_t> inDims = model.inputs[0].dims; // kept alive for the call below
    Qnn_Tensor_t inTensor = QNN_TENSOR_INIT;
    inTensor.version = QNN_TENSOR_VERSION_1;
    inTensor.v1.name = model.inputs[0].name.c_str();
    inTensor.v1.dataType = model.inputs[0].dataType;
    inTensor.v1.rank = static_cast<uint32_t>(inDims.size());
    inTensor.v1.dimensions = inDims.data();
    inTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    inTensor.v1.clientBuf.data = inputData;
    inTensor.v1.clientBuf.dataSize =
        model.inputs[0].elementCount() * static_cast<uint32_t>(elementBytes(model.inputs[0].dataType));

    // --- Allocate fresh output buffers (owned here, freed at the end of this call) and build
    // their Qnn_Tensor_t entries. ---
    std::vector<std::vector<uint32_t>> outDimsStorage(model.outputs.size());
    std::vector<std::vector<uint8_t>> outBuffers(model.outputs.size());
    std::vector<Qnn_Tensor_t> outTensors(model.outputs.size());
    for (size_t i = 0; i < model.outputs.size(); i++) {
        const auto& spec = model.outputs[i];
        outDimsStorage[i] = spec.dims;
        size_t bytes = spec.elementCount() * elementBytes(spec.dataType);
        outBuffers[i].resize(bytes);

        Qnn_Tensor_t t = QNN_TENSOR_INIT;
        t.version = QNN_TENSOR_VERSION_1;
        t.v1.name = spec.name.c_str();
        t.v1.dataType = spec.dataType;
        t.v1.rank = static_cast<uint32_t>(outDimsStorage[i].size());
        t.v1.dimensions = outDimsStorage[i].data();
        t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        t.v1.clientBuf.data = outBuffers[i].data();
        t.v1.clientBuf.dataSize = static_cast<uint32_t>(bytes);
        outTensors[i] = t;
    }

    Qnn_ErrorHandle_t execErr = g_iface->graphExecute(
        model.graph, &inTensor, 1,
        outTensors.data(), static_cast<uint32_t>(outTensors.size()),
        nullptr, nullptr);
    if (execErr != QNN_SUCCESS) {
        LOGE("graphExecute failed for %s (err=%lld)", modelKey.c_str(), (long long) execErr);
        return env->NewObjectArray(0, objClass, nullptr);
    }

    // --- Convert to the (IntArray shape, FloatArray data) pairs NativeQnnBackend.kt expects.
    // Data is copied out as float32 regardless of the tensor's on-wire dataType (our exports
    // are all "float" precision per the .bin's own metadata.json, so this is a straight
    // reinterpret, not a real dequantize — a quantized model would need real conversion here). ---
    jobjectArray result = env->NewObjectArray(
        static_cast<jsize>(model.outputs.size()) * 2, objClass, nullptr);
    for (size_t i = 0; i < model.outputs.size(); i++) {
        const auto& spec = model.outputs[i];
        jintArray shapeArr = env->NewIntArray(static_cast<jsize>(spec.dims.size()));
        std::vector<jint> shapeJ(spec.dims.begin(), spec.dims.end());
        env->SetIntArrayRegion(shapeArr, 0, static_cast<jsize>(shapeJ.size()), shapeJ.data());

        uint32_t n = spec.elementCount();
        jfloatArray dataArr = env->NewFloatArray(static_cast<jsize>(n));
        if (spec.dataType == QNN_DATATYPE_FLOAT_32) {
            env->SetFloatArrayRegion(
                dataArr, 0, static_cast<jsize>(n),
                reinterpret_cast<jfloat*>(outBuffers[i].data()));
        } else {
            LOGE("nativeRun(%s): output '%s' dataType 0x%x is not FLOAT_32 — "
                 "returning zeros, real conversion not implemented",
                 modelKey.c_str(), spec.name.c_str(), spec.dataType);
            std::vector<jfloat> zeros(n, 0.0f);
            env->SetFloatArrayRegion(dataArr, 0, static_cast<jsize>(n), zeros.data());
        }

        env->SetObjectArrayElement(result, static_cast<jsize>(i * 2), shapeArr);
        env->SetObjectArrayElement(result, static_cast<jsize>(i * 2 + 1), dataArr);
    }
    return result;
}

JNIEXPORT void JNICALL
Java_ai_secondsense_app_inference_qnn_NativeQnnBackend_nativeClose(
    JNIEnv* /*env*/, jobject /*thiz*/) {
    for (auto& [key, model] : g_models) {
        if (g_iface && model.context) {
            g_iface->contextFree(model.context, nullptr);
        }
    }
    g_models.clear();
    if (g_iface) {
        if (g_deviceHandle) g_iface->deviceFree(g_deviceHandle);
        if (g_backendHandle) g_iface->backendFree(g_backendHandle);
    }
    g_deviceHandle = nullptr;
    g_backendHandle = nullptr;
    if (g_backendLib) {
        dlclose(g_backendLib);
        g_backendLib = nullptr;
    }
    if (g_systemLib) {
        dlclose(g_systemLib);
        g_systemLib = nullptr;
        g_sysCreate = nullptr;
        g_sysGetBinaryInfo = nullptr;
        g_sysFree = nullptr;
    }
    g_iface = nullptr;
    g_ready = false;
    LOGI("QNN backend closed");
}

} // extern "C"
