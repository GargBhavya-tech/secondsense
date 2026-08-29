# QNN native bridge — build.gradle.kts only points CMake at this file when
# -PenableQnnNative=true is passed (see the externalNativeBuild block there). That keeps the
# default `gradlew assembleDebug` on the working TFLite path green even though this file
# exists on disk.
#
# REQUIRES: the real Qualcomm QNN SDK unpacked somewhere and pointed to via QNN_SDK_ROOT,
# either as a CMake -D flag (wired from gradle.properties' qnn.sdk.root) or an environment
# variable of the same name. Get it from https://qpm.qualcomm.com (Qualcomm Package Manager)
# or the Qualcomm AI Engine Direct SDK download page — this needs a Qualcomm account, which I
# cannot create on your behalf.
#
# Expected SDK layout (matches the standard QNN SDK zip):
#   $QNN_SDK_ROOT/include/QNN/            <- QnnInterface.h, QnnBackend.h, QnnGraph.h, etc.
#   $QNN_SDK_ROOT/lib/aarch64-android/    <- libQnnHtp.so, libQnnSystem.so, etc.
cmake_minimum_required(VERSION 3.22.1)
project(secondsense_qnn_bridge)

if(NOT DEFINED QNN_SDK_ROOT)
    if(DEFINED ENV{QNN_SDK_ROOT})
        set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
    else()
        message(FATAL_ERROR
            "QNN_SDK_ROOT not set. Pass -DQNN_SDK_ROOT=<path-to-unpacked-QNN-SDK> "
            "(wired automatically from gradle.properties' qnn.sdk.root, or set the "
            "QNN_SDK_ROOT environment variable) before building with -PenableQnnNative=true.")
    endif()
endif()

message(STATUS "QNN bridge: using SDK at ${QNN_SDK_ROOT}")

add_library(secondsense_qnn SHARED
    qnn_backend_jni.cpp
)

target_include_directories(secondsense_qnn PRIVATE
    ${QNN_SDK_ROOT}/include/QNN
)

# The QNN backend .so (libQnnHtp.so for the Hexagon HTP) is dlopen()'d at RUNTIME by
# qnn_backend_jni.cpp — NOT link-time linked — because Qualcomm ships several interchangeable
# backend .so files (HTP/CPU/GPU) selected by name at runtime, and only the one actually
# bundled into jniLibs needs to be present on a given device. We only need the SDK's headers
# at compile time; log + android are the only real link-time deps.
find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(secondsense_qnn
    ${log-lib}
    ${android-lib}
)

target_compile_options(secondsense_qnn PRIVATE -Wall -Wextra -fexceptions)
