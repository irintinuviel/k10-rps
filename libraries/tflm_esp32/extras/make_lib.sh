#!/bin/bash
# Builds an accelerated variant of the Arduino library tflm_esp32 2.0.0 for ESP32-S3:
#   - same TensorFlow Lite Micro sources as the original (compiled from source instead of the precompiled .a)
#   - CONV_2D and DEPTHWISE_CONV_2D use Espressif's ESP-NN routines (esp-tflite-micro e0159b2, esp-nn 34e9713)
# usage: make_lib.sh <original tflm_esp32 dir> <esp-tflite-micro clone> <esp-nn clone> <output dir>
set -euo pipefail
ORIG=$1; TFM=$2; NN=$3; OUT=$4
TFM_REV=e0159b2; NN_REV=34e9713
rm -rf "$OUT"; cp -r "$ORIG" "$OUT"
sed -i '/^precompiled=/d' "$OUT/library.properties"
sed -i 's/^sentence=.*/sentence=TensorFlow for ESP32 (tflm_esp32 2.0.0 + ESP-NN convolution kernels for ESP32-S3)/' "$OUT/library.properties"
rm -rf "$OUT/src/esp32s3"

# 1. ESP-NN versions of the two kernels that take nearly all of the time
K="$OUT/src/tensorflow/lite/micro/kernels"
for k in conv depthwise_conv; do
  { echo "// ESP-NN version of this kernel, from espressif/esp-tflite-micro $TFM_REV (tensorflow/lite/micro/kernels/esp_nn/$k.cc)"
    echo '#pragma GCC optimize ("O2")'
    echo '#define ESP_NN 1'
    git -C "$TFM" show "$TFM_REV:tensorflow/lite/micro/kernels/esp_nn/$k.cc"; } > "$K/$k.cpp"
done

# 2. ESP-NN itself: headers at the top of src/ (Arduino adds only src/ to the include path), sources below src/esp_nn/
TMP=$(mktemp -d); git -C "$NN" archive "$NN_REV" include src LICENSE | tar -x -C "$TMP"
cp "$TMP"/include/*.h "$OUT/src/"; cp "$TMP/src/common/common_functions.h" "$OUT/src/"
mkdir -p "$OUT/src/esp_nn"; cp -r "$TMP"/src/* "$OUT/src/esp_nn/"; cp "$TMP/LICENSE" "$OUT/src/esp_nn/LICENSE"
rm "$OUT/src/esp_nn/common/common_functions.h"; rm -rf "$TMP"
for f in $(find "$OUT/src/esp_nn" -name '*.c'); do sed -i '1i #pragma GCC optimize ("O2")' "$f"; done
# select the ESP32-S3 routines (in ESP-IDF this comes from Kconfig)
sed -i '0,/#pragma once/s//#pragma once\n#include "sdkconfig.h"\n#ifndef CONFIG_NN_OPTIMIZED\n#define CONFIG_NN_OPTIMIZED 1\n#endif/' "$OUT/src/esp_nn.h"

# 3. The original ships its audio "signal" sources twice (top level of src/ and src/signal/src/); the precompiled
#    .a hid that, a source build does not link. Keep the copies under src/signal/.
( cd "$OUT/src" && for f in *.cc; do [ -f "signal/src/${f%.cc}.cpp" ] && rm "$f"; done; rm -rf kiss_fft_wrappers )

# 4. Provenance note
mkdir -p "$OUT/extras"; cp "$0" "$OUT/extras/make_lib.sh"
cat > "$OUT/extras/README.txt" <<'TXT'
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
TXT
echo "library written to $OUT"
