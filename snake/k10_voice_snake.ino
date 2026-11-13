// K10 Voice Snake
// Gra Snake na UNIHIKER K10 sterowana głosem: "up", "down", "left", "right".
// Model keyword spotting (CNN int8, Speech Commands v2) działa w całości na płytce.
//
// Wymagania:
//  - Płytka "UNIHIKER K10" (Board Manager, URL: https://downloadcd.dfrobot.com.cn/UNIHIKER/package_unihiker_index.json)
//  - Biblioteka "MicroTFLite" (Library Manager)
//  - Plik model_data.h wygenerowany w Colabie (train_kws_k10.ipynb), skopiowany obok tego .ino
//
// Sterowanie:
//  - głos: up / down / left / right
//  - przycisk A: start / pauza / restart po przegranej
//  - przycisk B: zmiana prędkości węża

#include "unihiker_k10.h"
#include "driver/i2s.h"
#include "esp_heap_caps.h"
#include <MicroTFLite.h>

#include "audio_features.h"
#include "model_data.h"

// ===================== Ustawienia =====================
#define MIC_CHANNEL          0       // 0 = lewy kanał I2S (jak w przykładach K10), 1 = prawy
#define MIC_GAIN             1.0f    // dopasuj po obejrzeniu "peak" w Serial Monitorze (patrz README)
#define MIN_PEAK             700     // jeśli szczyt w oknie 1 s mniejszy -> cisza, bez inferencji
#define CONF_THRESHOLD       0.70f   // minimalna uśredniona pewność komendy
#define INFER_EVERY_SAMPLES  3200    // inferencja co 200 ms
#define AVG_WINDOW           3       // uśrednianie z ostatnich N inferencji
#define COMMAND_COOLDOWN_MS  700     // blokada powtórnej komendy
#define TENSOR_ARENA_SIZE    (48 * 1024)  // model zużywa ~23 KB
#define DEBUG_SERIAL         1

// ===================== Kontrola zgodności z notebookiem =====================
static_assert(MODEL_FEAT_SAMPLE_RATE == KWS_SAMPLE_RATE, "Inne SAMPLE_RATE niz w notebooku");
static_assert(MODEL_FEAT_CLIP_SAMPLES == KWS_CLIP_SAMPLES, "Inne CLIP niz w notebooku");
static_assert(MODEL_FEAT_FRAME_LEN == KWS_FRAME_LEN, "Inne FRAME_LEN niz w notebooku");
static_assert(MODEL_FEAT_FRAME_HOP == KWS_FRAME_HOP, "Inne FRAME_HOP niz w notebooku");
static_assert(MODEL_FEAT_FFT_SIZE == KWS_FFT_SIZE, "Inne FFT_SIZE niz w notebooku");
static_assert(MODEL_FEAT_NUM_MELS == KWS_NUM_MELS, "Inne N_MELS niz w notebooku");
static_assert(MODEL_FEAT_FMIN == KWS_FMIN, "Inne FMIN niz w notebooku");
static_assert(MODEL_FEAT_FMAX == KWS_FMAX, "Inne FMAX niz w notebooku");

// ===================== Kolory =====================
#define COLOR_BLACK   0x000000
#define COLOR_WHITE   0xFFFFFF
#define COLOR_GRAY    0x303030
#define COLOR_GREEN   0x00C853
#define COLOR_LIME    0x76FF03
#define COLOR_RED     0xFF1744
#define COLOR_YELLOW  0xFFD600
#define COLOR_BLUE    0x2962FF

UNIHIKER_K10 k10;
KwsFeatures featureExtractor;

// Globalne tensory z MicroTFLite.cpp. Czytamy/zapisujemy je bezpośrednio, bo ModelGetOutput()
// sprawdza tylko dims[1], a wyjście modelu ma kształt [1,1,1,6].
extern TfLiteTensor* tflInputTensor;
extern TfLiteTensor* tflOutputTensor;

// ===================== Audio (zadanie na rdzeniu 0) =====================
static int16_t* ring = nullptr;            // bufor kołowy 1 s
static volatile uint32_t ringHead = 0;     // indeks następnego zapisu
static volatile uint32_t totalSamples = 0; // licznik wszystkich próbek
static portMUX_TYPE ringMux = portMUX_INITIALIZER_UNLOCKED;

static void audioTask(void*) {
  static int16_t buf[512];  // 256 ramek stereo
  for (;;) {
    size_t bytesRead = 0;
    if (i2s_read(I2S_NUM_0, (uint8_t*)buf, sizeof(buf), &bytesRead, portMAX_DELAY) != ESP_OK) continue;
    const int frames = bytesRead / (2 * sizeof(int16_t));
    portENTER_CRITICAL(&ringMux);
    uint32_t head = ringHead;
    for (int i = 0; i < frames; i++) {
      float s = buf[2 * i + MIC_CHANNEL] * MIC_GAIN;
      if (s > 32767.0f) s = 32767.0f;
      if (s < -32768.0f) s = -32768.0f;
      ring[head] = (int16_t)s;
      head = (head + 1) % KWS_CLIP_SAMPLES;
    }
    ringHead = head;
    totalSamples += frames;
    portEXIT_CRITICAL(&ringMux);
  }
}

// ===================== Model =====================
static uint8_t* tensorArena = nullptr;
static int16_t* clip = nullptr;
static float* features = nullptr;
static float probHist[AVG_WINDOW][MODEL_NUM_LABELS];
static int histCount = 0, histPos = 0;
static uint32_t lastInferAt = 0;
static int idxSilence = -1, idxUp = -1, idxDown = -1, idxLeft = -1, idxRight = -1;

static void* allocPsramFirst(size_t n) {
  void* p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  return p ? p : malloc(n);
}

static int labelIndex(const char* name) {
  for (int i = 0; i < MODEL_NUM_LABELS; i++) if (strcmp(kModelLabels[i], name) == 0) return i;
  return -1;
}

// Zwraca peak sygnału w ostatniej sekundzie; probs wypełnione wynikiem modelu
static int runInference(float* probs) {
  portENTER_CRITICAL(&ringMux);
  const uint32_t head = ringHead;
  memcpy(clip, ring + head, (KWS_CLIP_SAMPLES - head) * sizeof(int16_t));
  memcpy(clip + (KWS_CLIP_SAMPLES - head), ring, head * sizeof(int16_t));
  portEXIT_CRITICAL(&ringMux);

  int peak = 0;
  for (int i = 0; i < KWS_CLIP_SAMPLES; i++) {
    int a = abs((int)clip[i]);
    if (a > peak) peak = a;
  }
  if (peak < MIN_PEAK) {  // za cicho - traktujemy jako ciszę
    for (int c = 0; c < MODEL_NUM_LABELS; c++) probs[c] = (c == idxSilence) ? 1.0f : 0.0f;
    return peak;
  }

  const uint32_t t0 = millis();
  featureExtractor.compute(clip, features);
  const float inScale = tflInputTensor->params.scale;
  const int inZp = tflInputTensor->params.zero_point;
  int8_t* in = tflInputTensor->data.int8;
  for (int i = 0; i < KWS_NUM_FEATURES; i++) {
    const float v = (features[i] - MODEL_FEATURE_MEAN) / MODEL_FEATURE_STD;
    int q = (int)lroundf(v / inScale) + inZp;
    in[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
  }
  const uint32_t t1 = millis();
  if (!ModelRunInference()) {
    Serial.println("[KWS] Blad inferencji");
    return peak;
  }
  const float outScale = tflOutputTensor->params.scale;
  const int outZp = tflOutputTensor->params.zero_point;
  for (int c = 0; c < MODEL_NUM_LABELS; c++) {
    probs[c] = (tflOutputTensor->data.int8[c] - outZp) * outScale;
  }

#if DEBUG_SERIAL
  int best = 0;
  for (int c = 1; c < MODEL_NUM_LABELS; c++) if (probs[c] > probs[best]) best = c;
  Serial.printf("[KWS] %-10s %.2f | peak %5d | cechy %lu ms, model %lu ms\n",
                kModelLabels[best], probs[best], peak, t1 - t0, millis() - t1);
#endif
  return peak;
}

// ===================== Snake =====================
#define COLS 12
#define ROWS 14
#define CELL 20
#define TOP  40   // pasek statusu u góry ekranu (240 x 320)

enum Dir { DIR_UP = 0, DIR_RIGHT, DIR_DOWN, DIR_LEFT };
struct Cell { int8_t x, y; };

static Cell snake[COLS * ROWS];
static int snakeLen = 0;
static int dir = DIR_RIGHT, nextDir = DIR_RIGHT;
static Cell food;
static bool gameOver = false, paused = true;
static int score = 0;
static uint32_t lastStep = 0, stepMs = 700;
static String lastWord = "-";
static int lastConf = 0;
static uint32_t ledOffAt = 0;
static bool needRender = true;

static void setLeds(uint8_t r, uint8_t g, uint8_t b, uint32_t ms) {
  if (k10.rgb == NULL) return;
  for (int i = 0; i < 3; i++) k10.rgb->write(i, r, g, b);
  ledOffAt = ms ? millis() + ms : 0;
}

static bool onSnake(int x, int y) {
  for (int i = 0; i < snakeLen; i++) if (snake[i].x == x && snake[i].y == y) return true;
  return false;
}

static void placeFood() {
  do { food.x = random(COLS); food.y = random(ROWS); } while (onSnake(food.x, food.y));
}

static void resetGame() {
  snakeLen = 3;
  for (int i = 0; i < snakeLen; i++) { snake[i].x = 5 - i; snake[i].y = ROWS / 2; }
  dir = nextDir = DIR_RIGHT;
  score = 0;
  gameOver = false;
  placeFood();
  needRender = true;
}

static void setDirection(int d) {
  if ((d + 2) % 4 != dir) nextDir = d;  // bez zawracania w miejscu
}

static void stepGame() {
  dir = nextDir;
  Cell head = snake[0];
  if (dir == DIR_UP) head.y--;
  if (dir == DIR_DOWN) head.y++;
  if (dir == DIR_LEFT) head.x--;
  if (dir == DIR_RIGHT) head.x++;
  // przechodzenie przez ściany - łatwiej przy opóźnieniu komend głosowych
  head.x = (head.x + COLS) % COLS;
  head.y = (head.y + ROWS) % ROWS;

  const bool eat = (head.x == food.x && head.y == food.y);
  // kolizja z ciałem (ogon się przesunie, więc go pomijamy, jeśli nie jemy)
  for (int i = 0; i < snakeLen - (eat ? 0 : 1); i++) {
    if (snake[i].x == head.x && snake[i].y == head.y) {
      gameOver = true;
      setLeds(255, 0, 0, 1500);
      needRender = true;
      return;
    }
  }
  if (eat && snakeLen < COLS * ROWS) snakeLen++;
  for (int i = snakeLen - 1; i > 0; i--) snake[i] = snake[i - 1];
  snake[0] = head;
  if (eat) {
    score++;
    setLeds(0, 255, 0, 250);
    if (snakeLen < COLS * ROWS) placeFood();
  }
  needRender = true;
}

static void render() {
  if (k10.canvas == NULL) return;
  k10.canvas->canvasClear();
  k10.canvas->canvasRectangle(0, 0, 240, 320, COLOR_BLACK, COLOR_BLACK, true);
  k10.canvas->canvasRectangle(0, 0, 240, TOP - 2, COLOR_GRAY, COLOR_GRAY, true);
  String status = lastWord + " " + String(lastConf) + "%  pkt:" + String(score);
  k10.canvas->canvasText(status, 6, 6, COLOR_WHITE, Canvas::eCNAndENFont24, 30, false);

  k10.canvas->canvasRectangle(food.x * CELL + 3, TOP + food.y * CELL + 3, CELL - 6, CELL - 6,
                              COLOR_RED, COLOR_RED, true);
  for (int i = snakeLen - 1; i >= 0; i--) {
    const uint32_t c = (i == 0) ? COLOR_LIME : COLOR_GREEN;
    k10.canvas->canvasRectangle(snake[i].x * CELL + 1, TOP + snake[i].y * CELL + 1, CELL - 2, CELL - 2,
                                c, c, true);
  }
  if (gameOver) {
    k10.canvas->canvasRectangle(20, 140, 200, 70, COLOR_RED, COLOR_BLACK, true);
    k10.canvas->canvasText("GAME OVER", 55, 150, COLOR_RED, Canvas::eCNAndENFont24, 20, false);
    k10.canvas->canvasText("A = restart", 60, 178, COLOR_WHITE, Canvas::eCNAndENFont24, 20, false);
  } else if (paused) {
    k10.canvas->canvasRectangle(20, 140, 200, 70, COLOR_YELLOW, COLOR_BLACK, true);
    k10.canvas->canvasText("PAUZA", 85, 150, COLOR_YELLOW, Canvas::eCNAndENFont24, 20, false);
    k10.canvas->canvasText("A = start", 70, 178, COLOR_WHITE, Canvas::eCNAndENFont24, 20, false);
  }
  k10.canvas->updateCanvas();
  needRender = false;
}

// ===================== Komendy =====================
static void handleCommand(int idx, float conf) {
  lastWord = kModelLabels[idx];
  lastWord.toUpperCase();
  lastConf = (int)(conf * 100.0f + 0.5f);
  if (idx == idxUp) { setDirection(DIR_UP); setLeds(0, 0, 255, 200); }
  else if (idx == idxDown) { setDirection(DIR_DOWN); setLeds(0, 0, 255, 200); }
  else if (idx == idxLeft) { setDirection(DIR_LEFT); setLeds(0, 0, 255, 200); }
  else if (idx == idxRight) { setDirection(DIR_RIGHT); setLeds(0, 0, 255, 200); }
  needRender = true;
}

static void processAudio() {
  const uint32_t now = totalSamples;
  if (now < KWS_CLIP_SAMPLES || now - lastInferAt < INFER_EVERY_SAMPLES) return;
  lastInferAt = now;

  float probs[MODEL_NUM_LABELS];
  runInference(probs);
  memcpy(probHist[histPos], probs, sizeof(probs));
  histPos = (histPos + 1) % AVG_WINDOW;
  if (histCount < AVG_WINDOW) histCount++;

  float avg[MODEL_NUM_LABELS] = {0};
  for (int h = 0; h < histCount; h++)
    for (int c = 0; c < MODEL_NUM_LABELS; c++) avg[c] += probHist[h][c] / histCount;

  int best = -1;
  const int cmds[4] = {idxUp, idxDown, idxLeft, idxRight};
  for (int i = 0; i < 4; i++) {
    if (cmds[i] >= 0 && (best < 0 || avg[cmds[i]] > avg[best])) best = cmds[i];
  }

  static uint32_t lastCmdMs = 0;
  if (best >= 0 && avg[best] >= CONF_THRESHOLD && millis() - lastCmdMs > COMMAND_COOLDOWN_MS) {
    lastCmdMs = millis();
    handleCommand(best, avg[best]);
    histCount = 0;  // to samo słowo nie wyzwoli się drugi raz
  }
}

// ===================== Przyciski =====================
static void handleButtons() {
  static bool prevA = false, prevB = false;
  const bool a = k10.buttonA->isPressed();
  const bool b = k10.buttonB->isPressed();
  if (a && !prevA) {
    if (gameOver) { resetGame(); paused = false; }
    else paused = !paused;
    needRender = true;
  }
  if (b && !prevB) {
    stepMs = (stepMs == 700) ? 450 : 700;
    setLeds(255, 214, 0, 200);
  }
  prevA = a; prevB = b;
}

// ===================== setup / loop =====================
void setup() {
  Serial.begin(115200);
  delay(500);

  k10.begin();
  k10.initScreen();
  k10.creatCanvas();
  k10.initI2S();
  i2s_set_clk(I2S_NUM_0, KWS_SAMPLE_RATE, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_STEREO);

  ring = (int16_t*)allocPsramFirst(KWS_CLIP_SAMPLES * sizeof(int16_t));
  clip = (int16_t*)allocPsramFirst(KWS_CLIP_SAMPLES * sizeof(int16_t));
  features = (float*)allocPsramFirst(KWS_NUM_FEATURES * sizeof(float));
  // arena w szybkiej pamięci wewnętrznej, a jeśli brak miejsca - w PSRAM
  tensorArena = (uint8_t*)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!tensorArena) tensorArena = (uint8_t*)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!ring || !clip || !features || !tensorArena) {
    Serial.println("Brak pamieci!");
    while (true) delay(1000);
  }
  memset(ring, 0, KWS_CLIP_SAMPLES * sizeof(int16_t));

  featureExtractor.begin();
  if (!ModelInit(g_model, tensorArena, TENSOR_ARENA_SIZE)) {
    Serial.println("ModelInit nie powiodl sie - zwieksz TENSOR_ARENA_SIZE");
    while (true) delay(1000);
  }
  if (tflInputTensor->type != kTfLiteInt8 || tflOutputTensor->type != kTfLiteInt8 ||
      tflInputTensor->bytes != KWS_NUM_FEATURES || tflOutputTensor->bytes != MODEL_NUM_LABELS) {
    Serial.printf("Niezgodny model: wejscie %u B, wyjscie %u B (oczekiwano %d i %d, int8)\n",
                  (unsigned)tflInputTensor->bytes, (unsigned)tflOutputTensor->bytes,
                  KWS_NUM_FEATURES, MODEL_NUM_LABELS);
    while (true) delay(1000);
  }
#if DEBUG_SERIAL
  ModelPrintTensorQuantizationParams();
#endif

  idxSilence = labelIndex("_silence_");
  idxUp = labelIndex("up");
  idxDown = labelIndex("down");
  idxLeft = labelIndex("left");
  idxRight = labelIndex("right");

  xTaskCreatePinnedToCore(audioTask, "audio", 4096, NULL, 5, NULL, 0);

  randomSeed(esp_random());
  resetGame();
  setLeds(0, 0, 0, 0);
  Serial.println("K10 Voice Snake gotowy. Przycisk A = start.");
}

void loop() {
  handleButtons();
  processAudio();

  if (!paused && !gameOver && millis() - lastStep >= stepMs) {
    lastStep = millis();
    stepGame();
  }
  if (ledOffAt && millis() > ledOffAt) setLeds(0, 0, 0, 0);
  if (needRender) render();
  delay(5);
}
