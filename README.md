# Kamień–papier–nożyce na UNIHIKER K10

Model rozpoznaje w obrazie z kamery K10 gest dłoni (papier, kamień, nożyce) albo to, że dłoni nie ma.
MobileNetV2 0.35, wejście 96×96, INT8, TensorFlow Lite Micro.

## Co tu jest

| Plik | Co to jest |
|---|---|
| `k10_rps.ino`, `model_data.h` | szkic na K10 i model, który trafia na płytkę (4 klasy: paper, rock, scissors, none) |
| `rps_int8.tflite`, `model.keras` | ten sam model jako plik TFLite i Keras |
| `rps_k10_colab.ipynb` | notebook, którym ten model został wytrenowany (Colab albo lokalnie) |
| `rps_k10_trening_bez_zdjec.ipynb` | wcześniejszy notebook (same sztuczne sceny) |
| `model_v2/` | wariant z 3 klasami, bez „none” |
| `libraries/tflm_esp32/` | biblioteka TensorFlow Lite Micro z przyspieszeniem dla ESP32-S3 |

## Jak zbudować szkic

1. Arduino IDE 2. W *Preferences → Additional boards manager URLs* dodaj
   `https://downloadcd.dfrobot.com.cn/UNIHIKER/package_unihiker_index.json`, w *Boards Manager* zainstaluj
   pakiet UNIHIKER (sprawdzone z wersją 0.0.5) i wybierz płytkę „unihiker k10”.
2. Skopiuj folder `libraries/tflm_esp32` do folderu bibliotek Arduino (Windows: `Dokumenty\Arduino\libraries`,
   Linux: `~/Arduino/libraries`). Jeśli jest tam już `tflm_esp32` z Menedżera bibliotek, zastąp go.
3. Skopiuj `k10_rps.ino` i `model_data.h` do folderu o nazwie `k10_rps` (Arduino wymaga, żeby folder
   nazywał się jak szkic), otwórz szkic i wgraj.
4. Żeby widzieć komunikaty szkicu w Serial Monitorze, ustaw *Tools → USB CDC On Boot: Enabled*.

Na Linuksie pakiet 0.0.5 nie buduje się od razu: szuka pliku `esptool.py`, a ma tylko program `esptool`.
Wystarczy dodać w `~/.arduino15/packages/UNIHIKER/tools/esptool_py/5.1.0/` plik `esptool.py`:

```python
#!/usr/bin/env python3
import os, sys
binary = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esptool")
os.execv(binary, [binary] + sys.argv[1:])
```

## Dlaczego własna kopia biblioteki

`tflm_esp32` 2.0.0 z Menedżera bibliotek nie ma przyspieszenia dla ESP32-S3: jedna predykcja tego modelu trwa
na K10 około 7,2 s. Kopia w `libraries/tflm_esp32` jest budowana ze źródeł i używa procedur ESP-NN firmy
Espressif w warstwach konwolucyjnych: ta sama predykcja trwa około 0,26 s. Szkic się nie zmienia.
Co dokładnie zostało zmienione i skąd pochodzą pliki, opisuje `libraries/tflm_esp32/extras/README.txt`.

Z tą biblioteką wyjścia modelu na K10 nie są identyczne co do bitu z wynikami na komputerze (ta sama klasa
na około 97% zdjęć testowych, trafność taka sama). Dlatego wbudowany „Test modelu”, który dopuszcza różnicę
najwyżej 8, może pokazać „ZLE!”, chociaż model działa poprawnie. Nagłówek z `rps_k10_colab.ipynb` nie zawiera
danych do tego testu, więc szkic pokazuje „Test modelu: brak danych”.

## Wyniki

Model INT8 na prawdziwych zdjęciach, których nie widział w treningu: dłonie osób z HaGRID spoza treningu
(2120 zdjęć), dłonie z osobnego zbioru zdjęć z kamerki (2131) oraz wycinki tych samych zdjęć, na których
dłoni nie ma (1672 i 103).

| Model | Dłonie: HaGRID | Dłonie: kamerka | Bez dłoni: odpowiedź „none” (HaGRID / kamerka) |
|---|---|---|---|
| `model_data.h` (4 klasy) | 92,2% | 73,5% | 90,7% / 88,3% |
| `model_v2/` (3 klasy) | 92,8% | 76,3% | nie ma klasy „none” |

Na samej płytce (1503 z tych zdjęć wysłane przez USB do modelu działającego na K10) wyniki zgadzają się
z komputerem z dokładnością do około jednego punktu procentowego.

Zbiory danych, z których korzysta trening, i ich licencje są opisane w `rps_k10_colab.ipynb`.
