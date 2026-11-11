/*
 * K10: kamień / papier / nożyce, model TFLite INT8 liczony na płytce
 * ----------------------------------------------------------------
 * Ekran: podgląd z kamery + rozpoznany gest, pewność i czas inferencji.
 * Przycisk A -> runda gry: odliczanie 3-2-1, K10 losuje swój ruch i pokazuje wynik.
 * Przycisk B -> zerowanie wyniku.
 *
 * Plik model_data.h wygeneruj notebookiem z folderu 1_trening i podmień go w tym folderze.
 * Płytka: Tools -> Board -> UNIHIKER K10 (pakiet płytek DFRobot UNIHIKER).
 */

#include "unihiker_k10.h"
#include <esp_camera.h>
#include <esp_heap_caps.h>
#include <esp_random.h>
#include <esp_system.h>

// TensorFlow Lite Micro: pakiet płytek oparty na arduino-esp32 3.x ma go wbudowanego
// (z akceleracją ESP-NN). Jeśli Twojego pakietu go nie ma, zainstaluj bibliotekę
// "tflm_esp32" z Menedżera bibliotek; wtedy zadziała linia poniżej.
#if !__has_include("tensorflow/lite/micro/micro_interpreter.h")
#include <tflm_esp32.h>
#endif
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

#include "model_data.h"   // g_model, g_model_len, MODEL_IMG, MODEL_LABELS, MODEL_NUM_CLASSES

// ----------------------------- ustawienia ---------------------------------
static const size_t ARENA_SIZE = 512 * 1024;    // pamięć robocza modelu w PSRAM (model zużywa ok. 250 KB)
static const float PROG_PEWNOSCI = 0.50f;       // poniżej tego progu pokazujemy "?"
static const float WYGLADZANIE = 0.5f;          // 0 = brak wygładzania, 0.9 = mocne
#define ZAMIEN_BAJTY_RGB565 0                   // kolejność bajtów RGB565, gdy auto-wykrywanie wyłączone
#define AUTO_KOLEJNOSC_BAJTOW 1                 // 1 = szkic sam sprawdza na pierwszych klatkach, która jest dobra
// ---------------------------------------------------------------------------

UNIHIKER_K10 k10;
extern SemaphoreHandle_t xLvglMutex;

// --- kamera i podgląd ---
static QueueHandle_t qCam = nullptr;
static uint8_t* podglad = nullptr;
static size_t podgladRozmiar = 0;
static lv_img_dsc_t podgladDsc;

// --- model ---
static uint8_t* arena = nullptr;
static tflite::MicroInterpreter* interp = nullptr;
static TfLiteTensor* wejscie = nullptr;
static TfLiteTensor* wyjscie = nullptr;
static float inScale = 1.0f;
static int inZero = 0;
static int8_t* bufor = nullptr;            // przygotowany obraz 96x96x3 czekający na model
static volatile bool buforGotowy = false;  // loop() -> zadanie inferencji
static SemaphoreHandle_t mWynik;
static float wynik[MODEL_NUM_CLASSES] = {0};
static volatile uint32_t czasMs = 0;

// --- interfejs i gra ---
static lv_obj_t* obraz;
static lv_obj_t* napisGora;
static lv_obj_t* napisDol;
static lv_obj_t* napisSrodek;
static lv_obj_t* miniatura;                  // "co widzi model": wejście modelu w prawdziwych kolorach
static lv_color_t* miniaturaBuf = nullptr;
static lv_img_dsc_t miniaturaDsc;
static char powodResetu[48] = "";
static volatile bool klikA = false, klikB = false;
enum Stan { NA_ZYWO, ODLICZANIE, WYNIK };
static Stan stan = NA_ZYWO;
static uint32_t stanOd = 0;
static int punktyTy = 0, punktyK10 = 0;
static char tekstWyniku[96];

void onA() { klikA = true; }
void onB() { klikB = true; }

// ============================== obraz -> model ==============================

static bool zamienBajty = ZAMIEN_BAJTY_RGB565;
static inline uint16_t pikselZ(const uint8_t* b, int i, bool zamien) {
  return zamien ? (uint16_t)(b[2 * i] | (b[2 * i + 1] << 8))      // little-endian
                : (uint16_t)((b[2 * i] << 8) | b[2 * i + 1]);     // big-endian (domyślne esp32-camera)
}
static inline uint16_t piksel(const uint8_t* b, int i) { return pikselZ(b, i, zamienBajty); }

// "Chropowatość" obrazu przy danej kolejności bajtów: przy złej kolejności sąsiednie piksele
// mają losowo skaczące kolory, więc suma różnic jest dużo większa. Dobra kolejność = gładszy obraz.
static uint32_t chropowatosc(const camera_fb_t* fb, bool zamien) {
  uint32_t suma = 0;
  const int W = fb->width, H = fb->height;
  for (int y = H / 4; y < 3 * H / 4; y += 8) {
    for (int x = W / 4; x < 3 * W / 4 - 1; x++) {
      uint16_t a = pikselZ(fb->buf, y * W + x, zamien), b = pikselZ(fb->buf, y * W + x + 1, zamien);
      suma += abs((int)(a >> 11) - (int)(b >> 11)) * 2 + abs((int)((a >> 5) & 0x3F) - (int)((b >> 5) & 0x3F))
            + abs((int)(a & 0x1F) - (int)(b & 0x1F)) * 2;
    }
  }
  return suma;
}
static uint32_t chropNorm = 0, chropZam = 0;
static int klatekBadania = 0;
static char opisBajtow[40] = "Bajty: sprawdzam...";

static inline int8_t kwantyzuj(float v) {
  int q = (int)lrintf(v / inScale) + inZero;
  return (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
}

// Środkowy kwadrat klatki -> MODEL_IMG x MODEL_IMG (uśrednianie bloków) -> INT8 RGB
static void przygotujWejscie(const camera_fb_t* fb, int8_t* dst) {
  const int W = fb->width, H = fb->height;
  const int S = W < H ? W : H;
  const int x0 = (W - S) / 2, y0 = (H - S) / 2;
  int k = 0;
  for (int oy = 0; oy < MODEL_IMG; oy++) {
    int ya = y0 + oy * S / MODEL_IMG, yb = y0 + (oy + 1) * S / MODEL_IMG;
    if (yb <= ya) yb = ya + 1;
    for (int ox = 0; ox < MODEL_IMG; ox++) {
      int xa = x0 + ox * S / MODEL_IMG, xb = x0 + (ox + 1) * S / MODEL_IMG;
      if (xb <= xa) xb = xa + 1;
      uint32_t r = 0, g = 0, b = 0, n = 0;
      for (int y = ya; y < yb; y++) {
        for (int x = xa; x < xb; x++) {
          uint16_t p = piksel(fb->buf, y * W + x);
          r += (p >> 11) & 0x1F;
          g += (p >> 5) & 0x3F;
          b += p & 0x1F;
          n++;
        }
      }
      dst[k++] = kwantyzuj(r * 255.0f / (31.0f * n));
      dst[k++] = kwantyzuj(g * 255.0f / (63.0f * n));
      dst[k++] = kwantyzuj(b * 255.0f / (31.0f * n));
    }
  }
}

// ============================ zadanie inferencji ============================

static void zadanieInferencji(void*) {
  const size_t rozmiar = MODEL_IMG * MODEL_IMG * 3;
  while (true) {
    if (!buforGotowy) { vTaskDelay(pdMS_TO_TICKS(5)); continue; }
    memcpy(wejscie->data.int8, bufor, rozmiar);
    buforGotowy = false;                       // loop() może już szykować kolejną klatkę

    uint32_t t0 = millis();
    if (interp->Invoke() != kTfLiteOk) { Serial.println("Invoke() nie powiodlo sie"); continue; }
    czasMs = millis() - t0;
    vTaskDelay(1);                             // oddaj na chwilę procesor innym zadaniom (watchdog)

    const float s = wyjscie->params.scale;
    const int z = wyjscie->params.zero_point;
    xSemaphoreTake(mWynik, portMAX_DELAY);
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
      float p = (wyjscie->data.int8[i] - z) * s;
      wynik[i] = WYGLADZANIE * wynik[i] + (1.0f - WYGLADZANIE) * p;
    }
    xSemaphoreGive(mWynik);
  }
}

static char blad[200] = "";
static char wynikTestu[48] = "Test modelu: brak danych";

// Liczy na K10 te same wejścia co komputer przy eksporcie i porównuje wyniki.
// OK = model działa na płytce jak w Colabie, więc ewentualne pomyłki biorą się z obrazu z kamery.
static void testModelu() {
#ifdef MODEL_TEST_COUNT
  int maxRoznica = 0, zleKlasy = 0;
  for (int t = 0; t < MODEL_TEST_COUNT; t++) {
    memcpy(wejscie->data.int8, MODEL_TEST_INPUT[t], MODEL_IMG * MODEL_IMG * 3);
    if (interp->Invoke() != kTfLiteOk) { snprintf(wynikTestu, sizeof(wynikTestu), "Test modelu: Invoke ZLE"); return; }
    int bK10 = 0, bPC = 0;
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
      int d = abs((int)wyjscie->data.int8[i] - (int)MODEL_TEST_OUTPUT[t][i]);
      if (d > maxRoznica) maxRoznica = d;
      if (wyjscie->data.int8[i] > wyjscie->data.int8[bK10]) bK10 = i;
      if (MODEL_TEST_OUTPUT[t][i] > MODEL_TEST_OUTPUT[t][bPC]) bPC = i;
    }
    Serial.printf("Test %d: K10 -> %s, komputer -> %s\n", t, MODEL_LABELS[bK10], MODEL_LABELS[bPC]);
    if (bK10 != bPC) zleKlasy++;
  }
  snprintf(wynikTestu, sizeof(wynikTestu), "Test modelu: %s (roznica %d)",
           (zleKlasy == 0 && maxRoznica <= 8) ? "OK" : "ZLE!", maxRoznica);
  Serial.println(wynikTestu);
#endif
}   // opis błędu startu, pokazywany na ekranie i w Serial Monitorze

static bool startModelu() {
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    snprintf(blad, sizeof(blad), "Zla wersja schematu modelu: %d (oczekiwana %d)",
             (int)model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  // Operacje użyte przez MobileNetV2 + głowicę, plus zapas na inne wersje konwertera TFLite
  static tflite::MicroMutableOpResolver<24> ops;
  ops.AddConv2D();
  ops.AddDepthwiseConv2D();
  ops.AddAdd();
  ops.AddMul();
  ops.AddMean();
  ops.AddSum();
  ops.AddFullyConnected();
  ops.AddSoftmax();
  ops.AddPad();
  ops.AddPadV2();
  ops.AddReshape();
  ops.AddQuantize();
  ops.AddDequantize();
  ops.AddAveragePool2D();
  ops.AddMaxPool2D();
  ops.AddRelu();
  ops.AddRelu6();
  ops.AddHardSwish();
  ops.AddLogistic();
  ops.AddConcatenation();
  ops.AddStridedSlice();
  ops.AddShape();
  ops.AddPack();
  ops.AddSqueeze();

  // Sprawdź, czy każda operacja z modelu jest obsługiwana, i nazwij te, których brakuje
  char brakujace[120] = "";
  auto* kody = model->operator_codes();
  for (unsigned i = 0; kody && i < kody->size(); i++) {
    tflite::BuiltinOperator op = tflite::GetBuiltinCode(kody->Get(i));
    const char* nazwa = tflite::EnumNameBuiltinOperator(op);
    Serial.printf("Operacja w modelu: %s (wersja %d)\n", nazwa, (int)kody->Get(i)->version());
    if (ops.FindOp(op) == nullptr) {
      size_t n = strlen(brakujace);
      snprintf(brakujace + n, sizeof(brakujace) - n, "%s%s", n ? ", " : "", nazwa);
    }
  }
  if (brakujace[0]) {
    snprintf(blad, sizeof(blad), "Brak obslugi operacji:\n%s", brakujace);
    return false;
  }

  size_t psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  Serial.printf("Wolny PSRAM: %u KB, wolny RAM wewn.: %u KB\n", (unsigned)(psram / 1024),
                (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
  if (psram == 0) {
    snprintf(blad, sizeof(blad), "PSRAM jest wylaczony.\nArduino IDE: Narzedzia -> PSRAM\n-> Enabled / OPI PSRAM");
    return false;
  }

  // próbujemy coraz większej areny; jeśli żadna nie wystarczy, problemem są operacje w modelu
  const size_t rozmiary[] = {ARENA_SIZE, 1024 * 1024, 2 * 1024 * 1024};
  for (size_t rozmiar : rozmiary) {
    if (rozmiar < ARENA_SIZE) continue;
    if (arena) { heap_caps_free(arena); arena = nullptr; }
    if (interp) { delete interp; interp = nullptr; }
    arena = (uint8_t*)heap_caps_aligned_alloc(16, rozmiar, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!arena) {
      snprintf(blad, sizeof(blad), "Brak PSRAM na arene %u KB (wolne %u KB)",
               (unsigned)(rozmiar / 1024), (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
      return false;
    }
    interp = new tflite::MicroInterpreter(model, ops, arena, rozmiar);
    if (interp->AllocateTensors() == kTfLiteOk) {
      blad[0] = 0;
      break;
    }
    snprintf(blad, sizeof(blad), "AllocateTensors() nie powiodlo sie\n(arena %u KB), operacje sa OK.\nZobacz Serial Monitor",
             (unsigned)(rozmiar / 1024));
    Serial.println(blad);
  }
  if (blad[0]) return false;

  wejscie = interp->input(0);
  wyjscie = interp->output(0);
  inScale = wejscie->params.scale;
  inZero = wejscie->params.zero_point;
  Serial.printf("Model OK: %u B, arena uzyta: %u B, wejscie %dx%dx%d, scale=%.4f zp=%d\n",
                g_model_len, (unsigned)interp->arena_used_bytes(),
                wejscie->dims->data[1], wejscie->dims->data[2], wejscie->dims->data[3],
                inScale, inZero);
  if (wejscie->type != kTfLiteInt8 || wejscie->dims->data[1] != MODEL_IMG) {
    snprintf(blad, sizeof(blad), "Model nie pasuje: wejscie typu %d, %dx%d\n(oczekiwane INT8 %dx%d)",
             (int)wejscie->type, wejscie->dims->data[1], wejscie->dims->data[2], MODEL_IMG, MODEL_IMG);
    return false;
  }
  testModelu();
  return true;
}

// ================================ gra =======================================

static const char* poPolsku(const char* etykieta) {
  if (!strcmp(etykieta, "rock")) return "KAMIEN";
  if (!strcmp(etykieta, "paper")) return "PAPIER";
  if (!strcmp(etykieta, "scissors")) return "NOZYCE";
  if (!strcmp(etykieta, "none")) return "BRAK REKI";
  return etykieta;
}

// 1 = a wygrywa, -1 = b wygrywa, 0 = remis
static int pojedynek(const char* a, const char* b) {
  if (!strcmp(a, b)) return 0;
  bool aWygrywa = (!strcmp(a, "rock") && !strcmp(b, "scissors")) ||
                  (!strcmp(a, "scissors") && !strcmp(b, "paper")) ||
                  (!strcmp(a, "paper") && !strcmp(b, "rock"));
  return aWygrywa ? 1 : -1;
}

static int najlepsza(float* pewnosc) {
  xSemaphoreTake(mWynik, portMAX_DELAY);
  int best = 0;
  for (int i = 1; i < MODEL_NUM_CLASSES; i++) if (wynik[i] > wynik[best]) best = i;
  *pewnosc = wynik[best];
  xSemaphoreGive(mWynik);
  return best;
}

static void rozstrzygnij() {
  float pew;
  int ty = najlepsza(&pew);
  if (pew < PROG_PEWNOSCI || !strcmp(MODEL_LABELS[ty], "none")) {
    snprintf(tekstWyniku, sizeof(tekstWyniku), "Nie widze gestu :(\nSprobuj jeszcze raz");
    return;
  }
  static const char* const RUCHY[3] = {"paper", "rock", "scissors"};
  const char* gracz = MODEL_LABELS[ty];
  const char* robot = RUCHY[esp_random() % 3];
  int w = pojedynek(gracz, robot);
  if (w > 0) punktyTy++;
  if (w < 0) punktyK10++;
  snprintf(tekstWyniku, sizeof(tekstWyniku), "Ty: %s\nK10: %s\n%s",
           poPolsku(gracz), poPolsku(robot), w > 0 ? "WYGRYWASZ!" : (w < 0 ? "K10 WYGRYWA" : "REMIS"));
}

// ============================== interfejs ===================================

static lv_obj_t* pasek(lv_align_t align) {
  lv_obj_t* l = lv_label_create(lv_scr_act());
  lv_obj_set_width(l, 240);
  lv_obj_set_style_bg_color(l, lv_color_black(), 0);
  lv_obj_set_style_bg_opa(l, LV_OPA_70, 0);
  lv_obj_set_style_text_color(l, lv_color_white(), 0);
  lv_obj_set_style_text_align(l, LV_TEXT_ALIGN_CENTER, 0);
  lv_obj_set_style_pad_all(l, 6, 0);
  lv_obj_align(l, align, 0, 0);
  return l;
}

static void odswiezNapisy() {
  char buf[160];
  float pew;
  int b = najlepsza(&pew);
  int n = 0;
  if (pew >= PROG_PEWNOSCI)
    n = snprintf(buf, sizeof(buf), "%s  %d%%   (%lu ms)\n", poPolsku(MODEL_LABELS[b]), (int)(pew * 100), (unsigned long)czasMs);
  else
    n = snprintf(buf, sizeof(buf), "?   (%lu ms)\n", (unsigned long)czasMs);
  xSemaphoreTake(mWynik, portMAX_DELAY);           // wszystkie prawdopodobieństwa: łatwiej ocenić model
  for (int i = 0; i < MODEL_NUM_CLASSES && n < (int)sizeof(buf); i++)
    n += snprintf(buf + n, sizeof(buf) - n, "%.3s %d%%  ", poPolsku(MODEL_LABELS[i]), (int)(wynik[i] * 100));
  xSemaphoreGive(mWynik);
  lv_label_set_text(napisGora, buf);

  if (millis() < 20000)
    snprintf(buf, sizeof(buf), "%s\n%s\n%s", powodResetu, wynikTestu, opisBajtow);
  else
    snprintf(buf, sizeof(buf), "Ty %d : %d K10\nA = zagraj   B = zeruj", punktyTy, punktyK10);
  lv_label_set_text(napisDol, buf);

  uint32_t t = millis() - stanOd;
  if (stan == ODLICZANIE) {
    snprintf(buf, sizeof(buf), "%d", 3 - (int)(t / 1000));
    lv_label_set_text(napisSrodek, buf);
    lv_obj_clear_flag(napisSrodek, LV_OBJ_FLAG_HIDDEN);
  } else if (stan == WYNIK) {
    lv_label_set_text(napisSrodek, tekstWyniku);
    lv_obj_clear_flag(napisSrodek, LV_OBJ_FLAG_HIDDEN);
  } else {
    lv_obj_add_flag(napisSrodek, LV_OBJ_FLAG_HIDDEN);
  }
}

// ================================ setup/loop ================================

static const char* opisResetu(esp_reset_reason_t r) {
  switch (r) {
    case ESP_RST_POWERON:  return "wlaczenie zasilania";
    case ESP_RST_SW:       return "reset programowy";
    case ESP_RST_PANIC:    return "BLAD PROGRAMU (panic)";
    case ESP_RST_INT_WDT:  return "WATCHDOG (przerwania)";
    case ESP_RST_TASK_WDT: return "WATCHDOG (zadanie)";
    case ESP_RST_WDT:      return "WATCHDOG";
    case ESP_RST_BROWNOUT: return "SPADEK NAPIECIA (kabel/USB)";
    case ESP_RST_DEEPSLEEP:return "wybudzenie";
    case ESP_RST_EXT:      return "przycisk reset";
    default:               return "inny";
  }
}

void setup() {
  Serial.begin(115200);
  esp_reset_reason_t rr = esp_reset_reason();
  snprintf(powodResetu, sizeof(powodResetu), "Start: %s", opisResetu(rr));
  Serial.println(powodResetu);
  disableCore0WDT();   // długie obliczenia modelu na rdzeniu 0 nie mogą wyzwalać watchdoga
  k10.begin();
  k10.initScreen();

  mWynik = xSemaphoreCreateMutex();
  bufor = (int8_t*)heap_caps_malloc(MODEL_IMG * MODEL_IMG * 3, MALLOC_CAP_8BIT);
  podgladRozmiar = 240 * 320 * 2;
  podglad = (uint8_t*)heap_caps_malloc(podgladRozmiar, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  xSemaphoreTake(xLvglMutex, portMAX_DELAY);
  lv_obj_set_style_bg_color(lv_scr_act(), lv_color_black(), 0);
  obraz = lv_img_create(lv_scr_act());
  lv_obj_set_pos(obraz, 0, 0);
  lv_obj_set_size(obraz, 240, 320);
  napisGora = pasek(LV_ALIGN_TOP_MID);
  napisDol = pasek(LV_ALIGN_BOTTOM_MID);
  napisSrodek = pasek(LV_ALIGN_CENTER);
#if LV_FONT_MONTSERRAT_28
  lv_obj_set_style_text_font(napisSrodek, &lv_font_montserrat_28, 0);
#endif
  lv_obj_add_flag(napisSrodek, LV_OBJ_FLAG_HIDDEN);
  miniaturaBuf = (lv_color_t*)heap_caps_malloc(MODEL_IMG * MODEL_IMG * sizeof(lv_color_t), MALLOC_CAP_8BIT);
  miniatura = lv_img_create(lv_scr_act());
  lv_obj_align(miniatura, LV_ALIGN_TOP_RIGHT, -4, 52);
  lv_obj_set_style_border_color(miniatura, lv_color_white(), 0);
  lv_obj_set_style_border_width(miniatura, 2, 0);
  lv_obj_add_flag(miniatura, LV_OBJ_FLAG_HIDDEN);
  lv_label_set_text_fmt(napisGora, "Ladowanie modelu...\n%s", powodResetu);
  lv_task_handler();
  xSemaphoreGive(xLvglMutex);

  bool ok = startModelu();
  if (ok && (!bufor || !podglad)) {
    snprintf(blad, sizeof(blad), "Brak pamieci na bufory obrazu\n(PSRAM wolne: %u KB)",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    ok = false;
  }
  if (!ok) {
    xSemaphoreTake(xLvglMutex, portMAX_DELAY);
    lv_label_set_text_fmt(napisGora, "BLAD:\n%s", blad);
    lv_task_handler();
    xSemaphoreGive(xLvglMutex);
    while (true) {                       // powtarzamy, żeby było widać po otwarciu Serial Monitora
      Serial.printf("BLAD: %s\n", blad);
      delay(2000);
    }
  }

  // inferencja na rdzeniu 0, a loop() (kamera + ekran) na rdzeniu 1
  xTaskCreatePinnedToCore(zadanieInferencji, "inferencja", 16 * 1024, nullptr, 1, nullptr, 0);

  qCam = xQueueCreate(2, sizeof(camera_fb_t*));
  register_camera(PIXFORMAT_RGB565, FRAMESIZE_QVGA, 2, qCam);

  k10.buttonA->setPressedCallback(onA);
  k10.buttonB->setPressedCallback(onB);
}

void loop() {
  camera_fb_t* fb = nullptr;
  if (!xQueueReceive(qCam, &fb, pdMS_TO_TICKS(200))) return;
  camera_fb_t* nowsza = nullptr;
  while (xQueueReceive(qCam, &nowsza, 0)) { esp_camera_fb_return(fb); fb = nowsza; }

#if AUTO_KOLEJNOSC_BAJTOW
  if (klatekBadania < 15) {                   // pierwsze klatki: ustal kolejność bajtów RGB565
    chropNorm += chropowatosc(fb, false);
    chropZam += chropowatosc(fb, true);
    if (++klatekBadania == 15) {
      zamienBajty = chropZam < chropNorm;
      snprintf(opisBajtow, sizeof(opisBajtow), "Bajty: %s (%lu/%lu)", zamienBajty ? "ZAMIENIONE" : "normalne",
               (unsigned long)(chropNorm / 1000), (unsigned long)(chropZam / 1000));
      Serial.println(opisBajtow);
    }
  }
#else
  snprintf(opisBajtow, sizeof(opisBajtow), "Bajty: %s (recznie)", zamienBajty ? "zamienione" : "normalne");
#endif
  if (!buforGotowy) {                         // model skończył poprzednią klatkę
    przygotujWejscie(fb, bufor);
    if (miniaturaBuf) {                         // podgląd wejścia modelu, kolory przez LVGL
      for (int i = 0; i < MODEL_IMG * MODEL_IMG; i++) {
        auto px = [&](int k) {
          int v = (int)lrintf((bufor[3 * i + k] - inZero) * inScale);
          return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        };
        miniaturaBuf[i] = lv_color_make(px(0), px(1), px(2));
      }
    }
    buforGotowy = true;
  }
  size_t n = fb->len < podgladRozmiar ? fb->len : podgladRozmiar;
  memcpy(podglad, fb->buf, n);
  int w = fb->width, h = fb->height;
  esp_camera_fb_return(fb);

  // logika gry
  if (klikB) { klikB = false; punktyTy = punktyK10 = 0; }
  if (klikA) { klikA = false; if (stan == NA_ZYWO) { stan = ODLICZANIE; stanOd = millis(); } }
  if (stan == ODLICZANIE && millis() - stanOd >= 3000) { rozstrzygnij(); stan = WYNIK; stanOd = millis(); }
  if (stan == WYNIK && millis() - stanOd >= 3000) { stan = NA_ZYWO; }

  xSemaphoreTake(xLvglMutex, portMAX_DELAY);
  podgladDsc.header.cf = LV_IMG_CF_TRUE_COLOR;
  podgladDsc.header.always_zero = 0;
  podgladDsc.header.w = w;
  podgladDsc.header.h = h;
  podgladDsc.data_size = n;
  podgladDsc.data = podglad;
  lv_img_set_src(obraz, &podgladDsc);
  lv_obj_invalidate(obraz);
  if (miniaturaBuf) {
    miniaturaDsc.header.cf = LV_IMG_CF_TRUE_COLOR;
    miniaturaDsc.header.always_zero = 0;
    miniaturaDsc.header.w = MODEL_IMG;
    miniaturaDsc.header.h = MODEL_IMG;
    miniaturaDsc.data_size = MODEL_IMG * MODEL_IMG * sizeof(lv_color_t);
    miniaturaDsc.data = (const uint8_t*)miniaturaBuf;
    lv_img_set_src(miniatura, &miniaturaDsc);
    lv_obj_clear_flag(miniatura, LV_OBJ_FLAG_HIDDEN);
    lv_obj_invalidate(miniatura);
  }
  odswiezNapisy();
  lv_task_handler();
  xSemaphoreGive(xLvglMutex);
}
