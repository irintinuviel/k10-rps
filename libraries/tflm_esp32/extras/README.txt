This folder is the Arduino library tflm_esp32 2.0.0 with two changes, made for the UNIHIKER K10 (ESP32-S3):

1. It is compiled from the sources that tflm_esp32 ships, not from its precompiled archive
   (library.properties no longer says "precompiled=full"; src/esp32s3/libtflm_esp32.a is removed).
2. The CONV_2D and DEPTHWISE_CONV_2D kernels use Espressif's ESP-NN routines for ESP32-S3:
   - src/tensorflow/lite/micro/kernels/conv.cpp and depthwise_conv.cpp come from
     espressif/esp-tflite-micro, commit e0159b2 (tensorflow/lite/micro/kernels/esp_nn/)
   - src/esp_nn/ and the esp_nn*.h / common_functions.h headers in src/ come from espressif/esp-nn, commit 34e9713
   Both are Apache-2.0 licensed (src/esp_nn/LICENSE).
Duplicate copies of the audio "signal" sources at the top of src/ were removed so that a source build links.

The original library has no ESP32-S3 acceleration. To go back to it, delete this folder and install
tflm_esp32 2.0.0 from the Arduino Library Manager. extras/make_lib.sh is the script that produced this folder.
