# Lekcja 24 — pełne podsumowanie
Podstawy sieci neuronowych: perceptron, aktywacje, softmax, MLP, metryki, early stopping.
---

## Zadanie 1

### Notatki własne

1. Perceptron = klasyfikator liniowy z funkcją progową.
2. Uczy się przez poprawianie wag, gdy się myli (reguła Rosenblatta).
3. Rozwiązuje problemy liniowo rozdzielne (AND, OR — tak; XOR — nie).
4. To protoplasta sieci neuronowych i regresji logistycznej.

Intuicja — co robi każda część?
Perceptron(n_features=2)     → tworzy model z 2 wagami + bias
fit(X, y)                    → uczy wag tak, by trafić w y
predict(X)                   → zwraca 0/1 dla nowych danych
activation(z >= 0 → 1)       → decyzja „tak/nie”
weights += lr * error * xi   → jak model się poprawia po błędzie


## Zadanie 2

### Notatki własne

1. Po co w ogóle funkcje aktywacji?
Bez nieliniowości wielowarstwowa sieć to w praktyce jedna funkcja liniowa — nie nauczy się XOR, złożonych wzorców itd. Aktywacja wprowadza nieliniowość i pozwala modelowi uczyć się skomplikowanych granic decyzyjnych.

2. Cztery funkcje z zadania 

- Sigmoid
(0, 1)
Warstwa wyjściowa: klasyfikacja binarna, prawdopodobieństwo
Intuicyjna interpretacja jako P(y=1)
Vanishing gradient — głębokie sieci uczą się wolno; wyjście nie jest wycentrowane

Tanh
(−1, 1)
Warstwy ukryte (rzadziej dziś)
Wycentrowana wokół 0 — często szybsza zbieżność niż sigmoid
Też vanishing gradient

ReLU
[0, ∞)
Domyślny wybór w warstwach ukrytych (CNN, MLP)
Szybka, prosta pochodna (0 lub 1), dobrze skaluje się w deep learning
Dying ReLU — neuron „umiera”, gdy ciągle dostaje ujemne wejście

Leaky ReLU
(−∞, ∞) (ujemna część z małym nachyleniem)
Warstwy ukryte, gdy martwe neurony są problemem
Ujemne wartości nie blokują uczenia całkowicie
Dodatkowy hiperparametr alpha

3. Pochodne — dlaczego są w zadaniu?
Backpropagation aktualizuje wagi przez łańcuch pochodnych. Na rozmowie wystarczy:
- Sigmoid: σ'(z) = σ(z) · (1 − σ(z))
- ReLU: 1 dla z > 0, 0 dla z ≤ 0
- Przy vanishing gradient sigmoid/tanh w głębokich sieciach gradient „zanika” → wolne uczenie → stąd popularność ReLU


4. 
- „Czym różni się sigmoid od ReLU?” → sigmoid: wygładzona, (0,1), do wyjścia; ReLU: prosta, szybka, standard w ukrytych warstwach.
- „Co to vanishing gradient?” → w głębokich sieciach z sigmoid/tanh gradient maleje warstwa po warstwie.
- „Co to dying ReLU?” → neuron zawsze dostaje ujemne wejście → wyjście 0 → gradient 0 → nie uczy się dalej.
- „Softmax vs sigmoid?” → sigmoid: jedna klasa binarna; softmax: wiele klas naraz (suma prawdopodobieństw = 1).

5.
W projekcie używasz sklearn, PyTorch lub TensorFlow — aktywacje wybierasz parametrem (activation='relu', nn.ReLU()). Zadanie uczy intuicji, nie codziennego kodowania wykresów.


## Zadanie 3

## Notatka — sigmoid vs softmax + interpretacja testów

### Sigmoid vs softmax — jaka jest różnica?

Oba zamieniają liczby z modelu na coś z zakresu 0–1, ale **służą do innych problemów**:

| | **Sigmoid** | **Softmax** |
|---|-------------|-------------|
| **Ile klas?** | 1 wyjście → **klasyfikacja binarna** (tak/nie) | Wiele wyjść → **klasyfikacja wieloklasowa** (A / B / C…) |
| **Wzór** | σ(z) = 1 / (1 + e^(-z)) | softmax(z_i) = e^(z_i) / Σ e^(z_j) |
| **Zakres** | Jedna liczba z (0, 1) | Wektor liczb z (0, 1) |
| **Suma** | Nie musi sumować się do 1 z innymi | **Suma wszystkich klas = 1** (pełny rozkład) |
| **Interpretacja** | P(y = klasa 1) | P(y = klasa i) dla każdej klasy |
| **Gdzie w sieci** | Warstwa wyjściowa — **2 klasy** | Warstwa wyjściowa — **3+ klasy** |
| **Przykład** | Czy email to spam? (0.87 = 87% szans na spam) | Iris: setosa / versicolor / virginica |

**Intuicja:** sigmoid odpowiada na pytanie *„jaka szansa na TAK?”*. Softmax dzieli 100% szans między **wszystkie możliwe klasy** — im wyższy logit danej klasy, tym większy jej udział w rozkładzie.

**Ważne:** softmax **wzmacnia różnice** między logitami. Mała przewaga w wejściu (np. 2 vs 1) daje umiarkowaną przewagę w prawdopodobieństwie (~66% vs ~24%). Duża przewaga (10 vs -10) daje niemal pewną predykcję (~99.99%).

---

### Interpretacja 3 testów z zadania

We wszystkich testach **najlepsza klasa to Klasa 0** (najwyższe prawdopodobieństwo), ale **pewność modelu** jest zupełnie inna:

#### Test 1: z = [2.0, 1.0, 0.1] — umiarkowana pewność

| Klasa | Prawdopodobieństwo |
|-------|-------------------|
| Klasa 0 | **65.9%** |
| Klasa 1 | 24.2% |
| Klasa 2 | 9.9% |

**Co to mówi:** Klasa 0 wygrywa, ale model **nie jest pewny**. Klasa 1 ma jeszcze ~24% szans — to typowa sytuacja, gdy logity są blisko siebie. W praktyce: predykcja = Klasa 0, ale warto sprawdzić, czy nie ma sensu pokazać użytkownikowi „drugą najlepszą” opcję.

#### Test 2: z = [1.0, 1.0, 1.0] — brak preferencji (maksymalna niepewność)

| Klasa | Prawdopodobieństwo |
|-------|-------------------|
| Klasa 0 | **33.3%** |
| Klasa 1 | **33.3%** |
| Klasa 2 | **33.3%** |

**Co to mówi:** Równe logity → **równy rozkład** (1/3 na klasę). Model nie ma podstaw, żeby wybrać jedną klasę — wszystkie są równie prawdopodobne. To przypadek **maksymalnej niepewności** softmaxa. W praktyce: technicznie wybieramy Klasę 0 (pierwsza z najwyższym wynikiem), ale **pewność predykcji jest minimalna**.

#### Test 3: z = [10.0, 0.0, -10.0] — niemal pewna predykcja

| Klasa | Prawdopodobieństwo |
|-------|-------------------|
| Klasa 0 | **99.995%** |
| Klasa 1 | 0.005% |
| Klasa 2 | ~0.000% |

**Co to mówi:** Ogromna różnica w logitach → softmax **prawie całe prawdopodobieństwo** daje Klasie 0. Klasy 1 i 2 są praktycznie wykluczone. To przypadek **najwyższej pewności** — model jest prawie pewien odpowiedzi. W praktyce: bezpiecznie można automatycznie przypisać Klasę 0 bez weryfikacji.

---

### Który test jest „najlepszy”?

To zależy, **co chcesz pokazać**:

| Test | Co najlepiej ilustruje |
|------|------------------------|
| **Test 2** | Równomierny rozkład — softmax przy **równych logitach** |
| **Test 1** | **Realistyczny** przypadek — wygrana klasa, ale z pewną niepewnością |
| **Test 3** | **Dominująca klasa** — softmax mocno „przykleja” się do jednej odpowiedzi |

**Podsumowanie wyników:** im większe różnice między logitami, tym bardziej „ostry” rozkład (Test 3). Im bardziej zbliżone logity, tym bardziej „miękki” i niepewny rozkład (Test 1). Równe logity → brak preferencji (Test 2).

---

## Zadanie 4

## Notatka — MLPClassifier na Iris + metryki i praca DS

### O czym jest to zadanie?

Trenujesz **gotową sieć neuronową** (`MLPClassifier` ze sklearn) na datasecie **Iris** — klasyfikacja 3 gatunków kwiatów na podstawie 4 cech (długość/szerokość płatków i działek).

To **nie** jest pisanie sieci od zera. To standardowy **pipeline ML**:

```
dane → podział train/test → skalowanie → model → predykcja → metryki → wykres loss
```

Architektura sieci: **4 wejścia → 10 neuronów → 5 neuronów → 3 klasy**.

---

### Co było liczone w zadaniu?

| Krok | Co robimy | Po co |
|------|-----------|-------|
| **Podział 80/20** | 80% uczymy model, 20% testujemy | Sprawdzamy, czy model **uogólnia**, a nie zapamiętuje |
| **StandardScaler** | Cechy w jednej skali (średnia 0, odch. 1) | MLP jest wrażliwy na skalę — bez tego uczenie bywa niestabilne |
| **MLPClassifier** | Sieć z 2 warstwami ukrytymi (10, 5), ReLU, Adam | Model uczy się granic między 3 klasami |
| **Accuracy** | % poprawnych predykcji na **teście** | Jedna liczba: „jak często trafił?” |
| **classification_report** | precision, recall, F1, support **per klasa** | Szczegóły: która klasa idzie gorzej |
| **Liczba epok (`n_iter_`)** | Ile razy model przeszedł przez dane treningowe | Czy uczenie doszło do końca |
| **Loss curve** | Błąd (cross-entropy) w kolejnych epokach | Czy model **się uczy** — loss powinien spadać |
| **Czas treningu** | Ile trwało `fit()` | Porównanie modeli / efektywność |

---

### Po co accuracy, precision, recall, F1, support?

| Metryka | Co mierzy | Pytanie, na które odpowiada |
|---------|-----------|----------------------------|
| **Accuracy** | Ogólny % trafień | „Jak dobrze model klasyfikuje **wszystko**?” |
| **Precision** | Trafność predykcji danej klasy | „Gdy model mówi *virginica*, to na pewno virginica?” |
| **Recall** | Ile prawdziwych przypadków klasy znalazł | „Czy znaleźliśmy **wszystkie** virginica?” |
| **F1-score** | Średnia harmoniczna precision i recall | „Jak model radzi sobie z tą klasą **ogółem**?” |
| **Support** | Liczba próbek danej klasy w teście | „Na ilu przykładach liczono te metryki?” (u nas: po 10) |

**Przykład z Iris (~97% accuracy):**

| Klasa | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| setosa | ~1.00 | ~1.00 | ~1.00 | 10 |
| versicolor | ~0.90 | ~1.00 | ~0.95 | 10 |
| virginica | ~1.00 | ~0.90 | ~0.95 | 10 |

**Interpretacja:**
- **setosa** — najłatwiejsza klasa, model trafia prawie idealnie
- **versicolor** — wysoki recall (znajduje wszystkie), nieco niższa precision (czasem myli z inną)
- **virginica** — wysoka precision (gdy mówi virginica, ma rację), recall ~0.90 (1 próbka pominięta)

**Ważne:** accuracy mówi „ogólnie dobrze”, ale **raport per klasa** pokazuje, gdzie model się myli — w pracy to często ważniejsze.

---

### Po co liczba epok i krzywa loss?

- **Epoka** = jeden pełny przejazd przez dane treningowe
- **`max_iter=500`** = maksymalnie 500 epok
- **`loss_curve_`** = jak spada błąd uczenia (cross-entropy)

| Sygnał na wykresie | Co to znaczy |
|--------------------|--------------|
| Loss **spada** | Model się uczy — OK |
| Loss **stoi w miejscu** | Może wystarczyło mniej epok / problem z hiperparametrami |
| Loss **rośnie** | Coś jest nie tak (złe LR, brak skalowania itd.) |

Krzywa loss to **diagnostyka treningu** — nie metryka biznesowa, ale bardzo praktyczna przy debugowaniu modelu.

---

### Czego uczy na przyszłą pracę DS?

**Tak — przydatne w codziennej pracy:**

1. **Pipeline klasyfikacji** — train/test, skaler, model, ewaluacja (standard w każdym projekcie)
2. **Czytanie `classification_report`** — często ważniejsze niż sama accuracy
3. **Skalowanie przed modelem** — reguła stosowana przy SVM, kNN, MLP, regresji
4. **Interpretacja wyników** — nie tylko „accuracy 97%”, ale „która klasa słabsza?”
5. **Idea MLP** — sieć wielowarstwowa; logika ewaluacji jest taka sama jak przy XGBoost czy Random Forest

**Nie — rzadko w pracy:**

- Pisanie backpropagation od zera
- Ręczne ustawianie wag neuronów
- Zapamiętywanie składni `MLPClassifier` na pamięć

---

### Czy musisz umieć napisać cały kod od zera?

**Na rozmowę:** nie linijka po linijce.

**Musisz umieć wytłumaczyć:**
- po co train/test i `stratify`
- po co `StandardScaler`
- co to accuracy, precision, recall, F1
- czym MLP różni się od drzewa / regresji logistycznej (elastyczniejszy, wolniejszy, wymaga skalowania)

**Szablon odpowiedzi na rozmowie (30 s):**

> „Trenowałam MLPClassifier na Iris: podział train/test, standaryzacja, model z dwiema warstwami ukrytymi. Oceniłam na teście accuracy i classification_report — precision, recall i F1 per klasa, bo sama accuracy nie mówi, która klasa jest problemem. Krzywa loss pokazuje, czy model się uczył. Na danych tabelarycznych częściej używam XGBoost lub regresji logistycznej, ale pipeline ewaluacji jest ten sam.”

---

### Czy w pracy będziesz pisać podobny kod?

**Tak, w podobnym duchu** — ten sam schemat:

```python
X_train, X_test, y_train, y_test = train_test_split(...)
scaler.fit_transform(X_train)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
classification_report(y_test, y_pred)
```

**Różnice w praktyce:**
- Częściej **XGBoost, LightGBM, Random Forest, Logistic Regression** niż MLP na danych tabelarycznych
- MLP / deep learning — głównie obrazy, tekst, audio
- Więcej: feature engineering, walidacja krzyżowa, tuning hiperparametrów
- Mniej: ręczne wykresy loss (chyba że deep learning)

**Podsumowanie:** Zadanie 4 uczy **całego procesu klasyfikacji i oceny modelu**, nie implementacji sieci. Kod ze sklearn będziesz pisać w podobnej formie; MLP to tu narzędzie do nauki, nie typowy wybór produkcyjny na takich danych.

---

## Zadanie 5

## Notatka — liczba parametrów w sieci + praca DS

### O czym jest to zadanie?

Obliczasz **ile parametrów (wag + biasów)** ma sieć neuronowa MLP dla różnych architektur — bez trenowania modelu, czysto matematycznie.

To odpowiedź na pytanie: *„Jak duża jest ta sieć?”* — im więcej parametrów, tym bardziej **złożony** model (większa pojemność, ale też ryzyko overfittingu i dłuższy trening).

---

### Co było liczone?

Dla każdej pary kolejnych warstw (np. 784 → 128):

| Składnik | Wzór | Przykład (784 → 128) |
|----------|------|----------------------|
| **Wagi** | `n_wejść × n_wyjść` | 784 × 128 = **100 352** |
| **Biasy** | `n_wyjść` (jeden na neuron) | **128** |
| **Razem warstwa** | wagi + biasy | **100 480** |

**Całkowita liczba parametrów** = suma po wszystkich warstwach.

Architektura `[784, 512, 256, 128, 64, 10]` oznacza:
```
784 wejść → 512 → 256 → 128 → 64 → 10 wyjść (klas)
```

---

### Wyniki 5 architektur z zadania — co mówią?

| Architektura | Parametry | Interpretacja |
|--------------|-----------|---------------|
| **(4, 3)** | 15 | Minimalna sieć (Iris: 4 cechy → 3 klasy, bez warstw ukrytych) |
| **(4, 10, 3)** | 83 | Mała sieć z 1 warstwą ukrytą — wystarczy na proste dane |
| **(784, 128, 10)** | ~102 tys. | Typowa sieć na MNIST (obraz 28×28 → 10 cyfr) |
| **(784, 256, 128, 10)** | ~235 tys. | Większa pojemność — więcej neuronów w ukrytych warstwach |
| **(784, 512, 256, 128, 64, 10)** | ~575 tys. | Duża sieć — 5 warstw ukrytych, parametry rosną **lawinowo** |

**Kluczowa obserwacja:** parametry rosną **nieliniowo** — szeroka pierwsza warstwa (784 → 512) daje najwięcej wag, bo mnożymy liczbę wejść × liczbę neuronów. Stąd w praktyce:
- nie robimy sieci „na wszelki wypadek” ogromnych
- zaczynamy od mniejszej architektury i zwiększamy, jeśli trzeba

**Wykres słupkowy** pokazuje skok z 83 parametrów do 100 tys.+ — dlatego na danych tabelarycznych częściej wybiera się prostsze modele (XGBoost, RF).

---

### Czy będziesz tego używać w pracy jako data scientist?

**Funkcji `count_parameters()` od zera — prawie nigdy.** W sklearn/PyTorch nie liczysz tego ręcznie w pętli.

**Samą koncepcję — tak, pośrednio:**

| W pracy | Jak to się łączy z zadaniem |
|---------|----------------------------|
| Wybór architektury / modelu | „Czy ten model nie jest za duży na nasze dane?” |
| Overfitting | Więcej parametrów → łatwiej zapamiętać train zamiast uczyć się wzorców |
| Czas treningu / koszt GPU | Większa sieć = wolniej i drożej |
| Regularyzacja (L2, dropout, early stopping) | Ograniczamy skutki zbyt dużej pojemności |
| Porównanie modeli | „Random Forest ma 500 drzew vs MLP ma 100 tys. parametrów” |

**Na danych tabelarycznych** (typowy DA/DS) rzadko trenujesz MLP w ogóle — częściej **XGBoost, LightGBM, Random Forest, regresja logistyczna**. Liczenie parametrów wraca głównie przy **deep learning** (obrazy, NLP).

---

### Czy musisz rozwiązać podobne zadanie na rozmowie?

**Pełne zadanie z barplotem — nie.**

**Możliwe pytania koncepcyjne:**

- *„Skąd biorą się parametry w sieci neuronowej?”* → wagi + biasy między warstwami
- *„Jak policzyć parametry między warstwą 100 a 50?”* → 100×50 wag + 50 biasów = **5 050**
- *„Co się stanie, gdy sieć ma za dużo parametrów?”* → overfitting, długi trening
- *„Dlaczego pierwsza warstwa ma najwięcej parametrów?”* → mnożymy przez liczbę cech wejściowych (np. 784 pikseli)

**Szablon odpowiedzi (30 s):**

> „Parametry w MLP to wagi i biasy. Na każde połączenie między warstwami przypada waga, plus jeden bias na neuron. Liczę to jako suma (n_wejść × n_wyjść + n_wyjść) dla każdej warstwy. Im szersza sieć, tym więcej parametrów — stąd na tabular data częściej wybieram prostsze modele, a duże sieci zostawiam pod obrazy i tekst.”

---

### Czy musisz umieć napisać cały kod od zera?

**Nie** — wystarczy znać wzór i umieć policzyć **jedną warstwę** na kartce.

W PyTorch sprawdzisz to jedną linijką: `sum(p.numel() for p in model.parameters())`.

---

### Podsumowanie

| Aspekt | Ocena |
|--------|-------|
| Kod `count_parameters()` w pracy | Rzadko |
| Intuicja „rozmiar modelu” | **Bardzo przydatna** |
| Wzór wagi + bias | Warto umieć wytłumaczyć |
| Podobne zadanie na rozmowie | Raczej pytanie teoretyczne, nie coding challenge |
| Barplot architektur | Edukacyjnie — pokazuje skok złożoności |

**Zadanie uczy myślenia o złożoności modelu**, nie codziennego liczenia parametrów. To ważne przy doborze architektury, regularyzacji i rozmowach o overfittingu — nawet jeśli w projekcie używasz gotowych bibliotek.

---

## Zadanie 6

## Notatka — porównanie aktywacji MLP + interpretacja wyników

### O czym jest to zadanie?

Porównujesz **3 funkcje aktywacji** w tej samej sieci MLP na datasecie **Breast Cancer** — jedyna zmienna to `activation`, reszta identyczna (architektura, solver, liczba epok).

Pytanie badawcze: *„Która aktywacja daje najlepszą klasyfikację na tym zbiorze?”*

To eksperyment **kontrolowany** — zmieniasz jeden element i obserwujesz wpływ na wynik.

---

### Co było liczone?

| Krok | Co robimy | Po co |
|------|-----------|-------|
| **`load_breast_cancer()`** | 569 próbek, 30 cech, 2 klasy (malignant / benign) | Większy i trudniejszy zbiór niż Iris |
| **`Pipeline(Scaler + MLP)`** | Skalowanie + model w jednym obiekcie | Skaler uczony **osobno w każdym foldzie CV** — bez wycieku danych |
| **3 modele MLP** | `activation='relu'`, `'tanh'`, `'logistic'` | Porównanie aktywacji przy tej samej architekturze (64, 32) |
| **`cross_val_score(..., cv=5)`** | 5 podziałów, accuracy w każdym | Stabilniejsza ocena niż jeden podział train/test |
| **Średnia ± std** | `scores.mean()` i `scores.std()` | Typowy wynik + rozrzut między foldami |
| **Bar chart** | Słupki = średnia accuracy, paski błędu = std | Szybkie porównanie wizualne |

**Ważne:** `logistic` w sklearn to **sigmoid** — ta sama funkcja, inna nazwa parametru.

---

### Co oznaczają wyniki?

Typowe wyniki po uruchomieniu (mogą minimalnie się różnić):

| Aktywacja | CV accuracy (średnia) | Std | Interpretacja |
|-----------|----------------------|-----|---------------|
| **relu** | ~**0.977** | ~0.009 | **Najlepsza** — standard w warstwach ukrytych |
| **logistic** | ~0.975 | ~0.003 | Bardzo blisko ReLU, **najmniejszy rozrzut** między foldami |
| **tanh** | ~0.967 | ~0.010 | Nieco słabsza, ale nadal >96% |

#### ReLU (~97.7%) — dlaczego wygrywa?
- Pochodna = 0 lub 1 → **szybsze uczenie**, mniej problemu z zanikającym gradientem
- Standard w deep learning i MLP na danych tabelarycznych
- Na Breast Cancer: model dobrze separuje klasy przy tej architekturze

#### Logistic / sigmoid (~97.5%) — co to znaczy?
- Wynik **prawie identyczny** z ReLU — na tym zbiorze wybór aktywacji ma **mały wpływ**
- Najniższe std (~0.003) → **najbardziej stabilny** wynik między foldami
- Historycznie popularna, dziś rzadziej w warstwach ukrytych (wolniejsza zbieżność w głębokich sieciach)

#### Tanh (~96.7%) — co to znaczy?
- Nieco gorsza średnia accuracy — różnica ~1 pp względem ReLU
- Wycentrowana wokół 0 (zakres −1 do 1) — kiedyś popularna, dziś ustępuje ReLU
- Na tym zadaniu: **działa dobrze**, ale nie jest najlepsza z trzech

---

### Jak czytać wykres słupkowy?

- **Wysokość słupka** = średnia accuracy z 5 foldów CV (im wyżej, tym lepiej)
- **Pasek błędu (cap)** = odchylenie standardowe — jak bardzo wyniki **różnią się** między foldami
  - Mały pasek (logistic) → stabilny wynik niezależnie od podziału danych
  - Większy pasek (tanh) → większa wrażliwość na to, które próbki trafiły do train/test
- **Oś Y ~0.9–1.0** — celowo „przybliżona”, żeby widać było różnice między aktywacjami (wszystkie są wysokie)

**Wniosek praktyczny:** różnice między aktywacjami są **niewielkie** (<2 pp) — na Breast Cancer z MLP (64, 32) każda z trzech aktywacji daje dobry wynik. ReLU wygrywa marginalnie.

---

### Dlaczego walidacja krzyżowa (cv=5), a nie jeden test?

| Podejście | Problem |
|-----------|---------|
| Jeden podział train/test | Wynik zależy od **szczęścia** przy podziale — raz 95%, raz 99% |
| **CV 5-fold** | 5 różnych podziałów → **średnia** i **std** dają wiarygodniejszy obraz |

W pracy: CV to standard przy **porównywaniu modeli** i **doborze hiperparametrów**.

---

### Czy będziesz tego używać w pracy DS?

**Porównywanie wariantów modelu — tak.** Ten sam schemat stosujesz przy:
- różnych algorytmach (XGBoost vs RF vs Logistic Regression)
- różnych hiperparametrach (głębokość drzewa, liczba neuronów)
- różnych zestawach cech

**Porównywanie aktywacji MLP — rzadko** na danych tabelarycznych (częściej używasz XGBoost/RF). Wraca to przy **deep learning** (obrazy, tekst).

**Pipeline + cross_val_score — bardzo często** — to wzorzec, który powtarza się w każdym projekcie ML.

---

### Szablon odpowiedzi na rozmowę (30 s)

> „Porównałam 3 aktywacje w MLP na Breast Cancer z walidacją krzyżową cv=5. Użyłam Pipeline ze StandardScaler, żeby skalowanie było poprawne w każdym foldzie. ReLU dało najwyższą średnią accuracy (~97.7%), tanh nieco gorsze (~96.7%). Różnice były małe — na tym zbiorze wybór aktywacji nie ma dramatycznego wpływu. Ważniejsze było użycie CV zamiast jednego podziału train/test.”

---

### Podsumowanie

| Element | Wniosek |
|---------|---------|
| **ReLU** | Najlepsza średnia accuracy — domyślny wybór w praktyce |
| **Logistic** | Prawie tak samo dobra, najstabilniejsza między foldami |
| **Tanh** | Nieco słabsza, ale nadal >96% |
| **CV vs jeden test** | CV daje wiarygodniejsze porównanie |
| **Pipeline** | Obowiązkowy przy skalowaniu + CV |
| **Praktyka DS** | Wzorzec „porównaj warianty przez CV” — uniwersalny; aktywacje MLP — niszowo |

---

## Zadanie 7

## Notatka — MLPRegressor na California Housing + praca DS

### O czym jest to zadanie?

Trenujesz **sieć neuronową do regresji** (`MLPRegressor`) — zamiast klasy (0/1/2) przewidujesz **liczbę ciągłą**: mediana ceny domu w California Housing.

To **Zadanie 4 w wersji regresyjnej** — ten sam typ modelu (MLP), inny typ problemu (regresja zamiast klasyfikacji).

```
cechy (dochód, wiek domu, współrzędne...) → MLP → przewidywana cena
```

---

### Co było liczone?

| Krok | Co robimy | Po co |
|------|-----------|-------|
| **`fetch_california_housing()`** | ~20 640 domów, 8 cech | Zbiór regresyjny — przewidywanie ceny |
| **Próbka 3000** | Losowy podzbiór (seed=42) | Szybszy trening (wymaganie z PDF) |
| **Podział 80/20** | Train / test | Ocena na danych niewidzianych |
| **`StandardScaler`** | Skalowanie cech | MLP wymaga podobnej skali cech (dobra praktyka) |
| **`MLPRegressor(32, 16)`** | 2 warstwy ukryte | Model nieliniowy — uczy złożonych zależności |
| **`early_stopping=True`** | Stop, gdy loss przestaje spadać | Unikamy przeuczenia i zbędnych epok |
| **RMSE, MAE, R²** | Metryki regresji na teście | Jak dobrze model trafia w cenę |
| **Wykres scatter** | Predykcje vs rzeczywiste | Wizualna ocena — czy punkty leżą na linii idealnej |
| **Loss curve** | Spadek błędu MSE w epokach | Diagnostyka treningu |

**Parametry modelu** — tylko z polecenia: `(32, 16)` i `early_stopping=True`. Reszta to domyślne wartości sklearn (`relu`, `adam`).

---

### Co oznaczają metryki?

Typowe wyniki po uruchomieniu:

| Metryka | Wartość | Co oznacza |
|---------|---------|------------|
| **RMSE** | ~0.59 | Średni błąd predykcji w **jednostkach ceny** (skala California Housing: ~0–5). Im niżej, tym lepiej |
| **MAE** | ~0.42 | Typowy błąd bezwzględny — „o tyle średnio się mylimy”. Mniej wrażliwy na outliers niż RMSE |
| **R²** | ~0.73 | Model wyjaśnia **~73% wariancji** cen. 1.0 = idealnie, 0 = nie lepszy od średniej |
| **Epoki** | ~130 | Early stopping zatrzymał trening — model przestał się poprawiać |

#### RMSE vs MAE — kiedy które?

| Metryka | Interpretacja | Kiedy ważniejsze |
|---------|---------------|------------------|
| **RMSE** | Kara za **duże błędy** (kwadrat) | Gdy duże pomyłki są kosztowne |
| **MAE** | Średni błąd **liniowy** | Gdy chcesz „typowy” błąd bez kara za outliers |
| **R²** | Jakość modelu **ogółem** | Porównanie modeli między sobą |

**Przykład:** RMSE 0.59 przy skali cen 0–5 oznacza, że model **całkiem dobrze** przewiduje ceny, ale nie idealnie — typowe dla MLP na tabular data bez tuningu.

#### R² ≈ 0.73 — co to znaczy?

- **73%** zmienności cen tłumaczy model
- **27%** to czynnik, którego model nie uchwycił (szum, brakujące cechy, ograniczenia MLP)
- W pracy: R² > 0.7 na regresji często uznaje się za **akceptowalny** wynik (zależy od dziedziny)

---

### Jak czytać wykresy?

#### 1. Predykcje vs rzeczywiste (scatter)

- **Czerwona linia przerywana** = idealna predykcja (y_pred = y_true)
- Punkty **blisko linii** → dobre predykcje
- Punkty **daleko od linii** → duże błędy (outliers)
- **Chmura wokół linii** → typowy rozrzut błędu

Jeśli chmura jest wąska wzdłuż linii — model działa dobrze. Jeśli „rozlana” — model słabo trafia.

#### 2. Krzywa loss (MSE)

- **Spadek na początku** → model się uczy
- **Płasko na końcu** → early stopping zatrzymał, gdy poprawa ustała
- **Loss (MSE)** w regresji = średni kwadrat błędu na zbiorze treningowym

---

### Klasyfikacja vs regresja — co się zmieniło?

| | Zadanie 4 (Iris) | Zadanie 7 (California) |
|---|------------------|------------------------|
| Model | `MLPClassifier` | `MLPRegressor` |
| Wyjście | Klasa (0/1/2) | Liczba (cena) |
| Metryki | accuracy, precision, recall | **RMSE, MAE, R²** |
| Funkcja loss | Cross-entropy | **MSE** (Mean Squared Error) |
| Wykres | Loss curve | Loss curve + **scatter pred vs true** |

---

### Czy będziesz tego używać w pracy DS?

**MLPRegressor na tabular data — rzadko.** Częściej:
- **Regresja liniowa**, **Ridge/Lasso** (lekcja 22)
- **XGBoost**, **Random Forest** — często lepsze na danych tabelarycznych

**Metryki RMSE, MAE, R² — tak, bardzo często** — przy każdym projekcie regresyjnym (ceny, prognozy, demand forecasting).

**Scatter pred vs true — tak** — standardowa wizualizacja jakości modelu regresyjnego.

**Early stopping — tak** — koncepcja wraca przy każdym modelu iteracyjnym (XGBoost, neural nets).

---

### Szablon odpowiedzi na rozmowę (30 s)

> „Trenowałam MLPRegressor na próbce California Housing — regresja, przewidywanie mediana ceny. Użyłam architektury (32, 16) z early stopping. Na teście: RMSE ~0.59, R² ~0.73 — model wyjaśnia ok. 73% wariancji. Scatter pokazał, że predykcje są blisko linii idealnej, ale z rozrzutem. W praktyce na tabular data częściej używam XGBoost lub regresji liniowej, ale metryki RMSE/MAE/R² stosuję zawsze.”

---

### Podsumowanie

| Element | Wniosek |
|---------|---------|
| **MLPRegressor** | Sieć neuronowa do regresji — ten sam rodzina co MLPClassifier |
| **RMSE / MAE / R²** | **Must know** — standard oceny regresji |
| **Scatter pred vs true** | Szybka wizualna ocena — stosuj w każdym projekcie regresyjnym |
| **Early stopping** | Przydatna technika — oszczędza czas, ogranicza overfitting |
| **Próbka 3000** | Szybszy eksperyment — w produkcji trenujesz na pełnych danych |
| **Praktyka DS** | Metryki i wykresy — tak; MLPRegressor na tabular — raczej nie |

---

## Zadanie 8

## Notatka — early stopping + interpretacja wyników

### O czym jest to zadanie?

Porównujesz **dwa sposoby zatrzymania treningu** tego samego MLP na Breast Cancer — jedyna różnica to `early_stopping=True` vs `False`.

Pytanie badawcze: *„Czy warto przerywać trening wcześniej — i co za to płacimy?”*

```
early_stopping=True  → stop, gdy model przestaje się poprawiać na walidacji
early_stopping=False → trenuj do max_iter (1000) lub pełnej zbieżności
```

---

### Co było liczone?

| Element | Co mierzymy | Po co |
|---------|-------------|-------|
| **Czas treningu** | `time.time()` przed/po `fit()` | Ile trwa uczenie — koszt obliczeniowy |
| **Liczba epok (`n_iter_`)** | Ile iteracji model faktycznie wykonał | Czy early stopping skrócił trening |
| **Accuracy na teście** | Trafność na 20% danych niewidzianych | Czy wcześniejsze zatrzymanie **pogorszyło** wynik |
| **Loss curve (oba warianty)** | Błąd w kolejnych epokach | Wizualnie: gdzie True się zatrzymuje vs False |

**Parametry z polecenia:** `(64, 32)`, `max_iter=1000` — reszta to domyślne sklearn.

---

### Co oznaczają typowe wyniki?

| | early_stopping=True | early_stopping=False |
|---|----------------------|----------------------|
| **Epoki** | ~20–40 | ~150–200 |
| **Czas** | ~0.02–0.06 s | ~0.10–0.15 s |
| **Accuracy** | ~0.94–0.97 | ~0.95–0.97 |

*(Dokładne liczby mogą minimalnie się różnić między uruchomieniami.)*

#### early_stopping=True — co to znaczy?

- Sklearn **odcina 10% danych treningowych** na walidację wewnętrzną
- Gdy loss na walidacji **przez 10 epok nie spada** → trening się kończy
- **Mniej epok, krócej** — oszczędność czasu i energii
- Accuracy **podobna lub nieco niższa** — czasem wcześniejsze stop „ucina” dalszą poprawę

#### early_stopping=False — co to znaczy?

- Model trenuje **dłużej** (aż do zbieżności lub 1000 epok)
- **Więcej epok, dłużej** — większy koszt obliczeniowy
- Accuracy **czasem wyższa** — model ma szansę dalej się uczyć
- Ryzyko: na trudniejszych danych długi trening → **overfitting** (dobry train, gorszy test)

#### Jak czytać wykres loss?

- **Niebieska linia (True)** — kończy się **wcześniej** (mniej epok)
- **Pomarańczowa linia (False)** — schodzi **niżej / dalej** — więcej epok uczenia
- Jeśli obie linie są blisko na końcu — early stopping zatrzymał w sensownym momencie
- Jeśli False schodzi wyraźnie niżej — model **mógł** jeszcze coś zyskać dłuższym treningiem

---

### Główny wniosek z zadania

| Aspekt | early_stopping=True | early_stopping=False |
|--------|---------------------|----------------------|
| **Szybkość** | ✅ Szybciej | ❌ Wolniej |
| **Epoki** | ✅ Mniej | ❌ Więcej |
| **Accuracy** | ≈ podobna | ≈ podobna lub nieco wyższa |
| **Overfitting** | ✅ Ogranicza ryzyko | ⚠️ Większe ryzyko |
| **Praktyka** | ✅ **Domyślny wybór** w produkcji | Rzadziej — gdy masz czas i kontrolujesz overfitting |

**Kompromis:** early stopping to **regularyzacja przez czas** — przerywasz, zanim model zacznie „wkuwać” szum. Na Breast Cancer różnica accuracy bywa mała, ale **oszczędność czasu jest duża** (np. 20 vs 180 epok).

---

### Po co early stopping w pracy DS?

**Tak — koncepcja jest bardzo praktyczna:**

| Gdzie | Zastosowanie |
|-------|--------------|
| **MLP / sieci neuronowe** | `early_stopping=True` — standard |
| **XGBoost** | `early_stopping_rounds` — ten sam pomysł |
| **LightGBM** | `early_stopping()` callback |
| **Deep learning (PyTorch/Keras)** | Early stopping + checkpoint najlepszego modelu |

Nie musisz implementować logiki ręcznie — biblioteki robią to za Ciebie. Ważne, żeby **wiedzieć, po co to włączasz**.

---

### Szablon odpowiedzi na rozmowę (30 s)

> „Porównałam MLP z early stopping włączonym i wyłączonym na Breast Cancer. Z early stopping trening trwał krócej i używał mniej epok, bo sklearn zatrzymuje model, gdy loss na walidacji przestaje spadać. Accuracy na teście była podobna — czasem nieco niższa przy True, ale oszczędność czasu jest duża. To forma regularyzacji — chroni przed overfittingiem. Ten sam pomysł stosuję w XGBoost przez early_stopping_rounds.”

---

### Podsumowanie

| Element | Wniosek |
|---------|---------|
| **Early stopping** | Przerwij trening, gdy model przestaje się poprawiać |
| **Czas vs epoki** | True = szybciej i mniej iteracji |
| **Accuracy** | Często podobna — nie zawsze warto trenować do końca |
| **Loss curve** | Wizualna diagnostyka — gdzie model się „nasycił” |
| **Praktyka DS** | Włączaj early stopping przy modelach iteracyjnych |
| **Kod od zera** | Nie — wystarczy znać parametr i intuicję |

---

## Zadanie 11

## Notatka — MLP vs Random Forest na make_moons

### O czym jest to zadanie?

Porównujesz **dwie różne klasy modeli** na tym samym zbiorze syntetycznym `make_moons`:
- **MLPClassifier (20, 10)** — sieć neuronowa, granica **gładka**
- **RandomForestClassifier (100 drzew)** — ensemble drzew, granica **poszarpana**

Pytanie badawcze: *„Jak różne modele radzą sobie z tym samym problemem nieliniowym — i czy widać to na wykresie?”*

```
make_moons → train/test → MLP (+ skalowanie) + RF → accuracy, czas → 2 wykresy contourf
```

---

### Co było liczone?

| Krok | Co robimy | Po co |
|------|-----------|-------|
| **`make_moons(500, noise=0.3)`** | 500 punktów w kształcie dwóch „księżyców” | Problem **nieliniowo rozdzielny** — test dla sieci i drzew |
| **Podział train/test 80/20** | Ocena na danych niewidzianych | Accuracy na teście, nie na train |
| **MLP `(20, 10)`** | 2 warstwy ukryte | Sieć uczy się zakrzywionej granicy |
| **`StandardScaler` dla MLP** | Skalowanie cech | MLP wymaga podobnej skali — dobra praktyka |
| **RF `n_estimators=100`** | 100 drzew decyzyjnych | Klasyk ML na danych tabelarycznych / nieliniowych |
| **Accuracy + czas** | `accuracy_score`, `time.time()` | Porównanie jakości i kosztu treningu |
| **`contourf` × 2** | Granice decyzyjne obok siebie | Wizualna różnica modeli — ważniejsza niż sama liczba accuracy |
| **Ręczny `StandardScaler`** | `fit` na train, `transform` na test | MLP wymaga skalowania; RF — surowe cechy |

**Parametry wyłącznie z polecenia:** `make_moons(n_samples=500, noise=0.3)`, MLP `(20, 10)`, RF 100 drzew.

---

### Pipeline czy ręczny StandardScaler?

W Zadaniu 11 **nie używasz `Pipeline`** — tak jest w PDF (sekcja 6.1). To świadomy wybór, nie błąd.

| Sytuacja | Co stosujesz | Dlaczego |
|----------|--------------|----------|
| **Z11 — jeden train/test + wykres** | Ręczny skaler: `fit(X_train)` → `transform(X_test)` | Wystarczy; kod czytelnie pokazuje: MLP = skalowane, RF = surowe |
| **Z6, Wine — `cross_val_score`** | `Pipeline([StandardScaler, MLP])` | W każdym foldzie skaler uczy się **tylko na trainie** — bez wycieku |

**Reguła:** Pipeline przy CV i `GridSearchCV`; przy pojedynczym podziale ręczny skaler jest OK, jeśli `fit` tylko na train.

Można by użyć Pipeline także w Z11 (`mlp_pipe.predict` na siatce w oryginalnych współrzędnych), ale PDF uczy krok po kroku — stąd osobny `StandardScaler`.

---

### Co oznaczają typowe wyniki?

| Model | Accuracy (test) | Czas | Granica decyzyjna |
|-------|-----------------|------|-------------------|
| **MLP (20, 10)** | ~0.87–0.92 | ~0.08–0.2 s | Gładka, zakrzywiona linia |
| **Random Forest (100)** | ~0.87–0.89 | ~0.07–0.1 s | Poszarpana, „schodkowa” |

#### Dlaczego make_moons?

- Perceptron (Zadanie 1) **nie rozwiązałby** tego problemu — dane nie są liniowo rozdzielne
- `noise=0.3` dodaje **szum** — punkty lekko „rozmyte”, granica nie jest idealna
- To **laboratoryjny** zbiór do pokazania, że MLP i RF uczą się nieliniowości

#### MLP — co widać na wykresie?

- Granica **ciągła i gładka** — charakterystyka sieci z funkcją aktywacji (ReLU domyślnie)
- Architektura `(20, 10)` = wystarczająco neuronów, by objąć kształt księżyców
- Skalowane cechy — trening stabilniejszy

#### Random Forest — co widać na wykresie?

- Granica **kanciasta** — każde drzewo dzieli przestrzeń prostymi cutami (prostopadłymi osiom)
- 100 drzew **uśrednia** głosowanie → poszarpana, ale skuteczna granica
- Nie wymaga skalowania — drzewa patrzą tylko na „lewo/prawo” względem progu

#### Podobna accuracy, inny model

- Oba modele mogą mieć **~87% accuracy**, ale **uczą inaczej**
- Metryka nie mówi wszystkiego — stąd **wykres granic decyzyjnych**
- W pracy: przy wyborze modelu patrzysz też na interpretowalność, czas, stabilność

---

### Jak czytać wykresy contourf?

- **Kolor tła** — przewidywana klasa w danym regionie (contourf)
- **Kropki** — prawdziwe punkty treningowe (czarne obwódki)
- **Granica kolorów** — linia decyzyjna modelu
- **MLP (lewy)** — płynna krzywa między klasami
- **RF (prawy)** — „pixelowany” lub schodkowy kształt

Jeśli oba modele dobrze klasyfikują, kolory tła zgadzają się z kropkami w większości obszaru; błędy widać przy granicy i w strefie szumu.

---

### MLP vs Random Forest — kiedy co w pracy?

| | MLP | Random Forest |
|---|-----|---------------|
| **Dane tabularne** | Rzadziej | **Często pierwszy wybór** |
| **Granica** | Gładka | Poszarpana |
| **Skalowanie** | Zwykle tak | Nie |
| **Interpretacja** | Trudniejsza | Feature importance |
| **Obrazy / tekst** | Tak (deep learning) | Nie |

Na **make_moons** oba działają — to ilustracja, nie typowy projekt biznesowy.

---

### Czy będziesz tego używać w pracy DS?

**`make_moons` bezpośrednio — nie.** To zbiór do nauki i demo.

**Porównywanie modeli + wizualizacja granic — tak, pośrednio:**
- Wybór między algorytmami (baseline RF vs prostszy model)
- Prezentacja klientowi / zespołowi „jak model dzieli przestrzeń”
- Debugowanie: czy model w ogóle łapie wzorzec

**Accuracy + czas treningu** — standard w każdym eksperymencie.

---

### Szablon odpowiedzi na rozmowę (30 s)

> „Na make_moons porównałam MLP (20, 10) z Random Forest (100 drzew). Oba osiągnęły podobną accuracy (~87%), ale granice decyzyjne wyglądają inaczej — MLP ma gładką krzywą, RF poszarpaną. make_moons to problem nieliniowy — jedna linia go nie rozdzieli, więc perceptron by nie wystarczył. W praktyce na tabular data częściej zaczynam od Random Forest lub XGBoost, a sieci stosuję przy obrazach i tekście.”

---

### Podsumowanie

| Element | Wniosek |
|---------|---------|
| **make_moons** | Problem nieliniowy — test dla MLP i ensemble |
| **MLP (20, 10)** | Gładka granica, wymaga skalowania |
| **RF (100 drzew)** | Poszarpana granica, bez skalowania |
| **Accuracy** | Często podobna — wykres pokazuje więcej niż sama metryka |
| **contourf** | Narzędzie do wizualizacji granic — przydatne w analizie |
| **Skalowanie** | MLP: ręczny StandardScaler; RF: surowe dane |
| **Pipeline** | Nie w Z11 — tak przy CV (Z6, Wine) |
| **Praktyka DS** | RF/XGBoost na tabular; MLP/deep learning — inne domeny |
| **Kod od zera** | Nie — ważna intuicja modeli i wizualizacja |
| **Pipeline** | Nie w Z11 — tak w zadaniach z CV (Z6, Wine) |

---

## Zadanie dodatkowe — Wine Quality

## Notatka — MLP vs Random Forest na Wine Quality

### O czym jest to zadanie?

Porównujesz **3 architektury MLP** z **Random Forest** na **danych tabelarycznych** — czerwone wino (UCI Wine Quality, ten sam plik co w lekcji 23). To zamknięte zadanie: wszystkie parametry są w tabeli polecenia (jak Z6, Z7, Z11).

Pytanie badawcze: *„Na małym datasetcie tabularnym — czy prosta sieć sklearn dorównuje Random Forest?”*

```
wczytaj CSV → binaryzuj target → 800 próbek → CV=5 (3 MLP + RF) → bar chart → test MLP (32,16) vs RF
```

---

### Co było liczone?

| Krok | Co robimy | Po co |
|------|-----------|-------|
| **`winequality-red.csv`** | 11 cech chemicznych + `quality` | Realistyczne dane tabularne z lekcji 23 |
| **Target binarny** | `quality >= 7` → 1 („good”) | Klasyfikacja binarna — rzadka klasa pozytywna (~14%) |
| **800 próbek** | `sample(n=800, random_state=42)` | Szybsze CV; pełny zbiór ma ~1599 wierszy |
| **3 architektury MLP** | `(32,)`, `(32, 16)`, `(64, 32)` | Mały / średni / duży — bez „szukania” najlepszego |
| **`Pipeline(Scaler + MLP)`** | Skalowanie + model | Poprawne CV — skaler w każdym foldzie osobno |
| **Wspólne parametry MLP** | `relu`, `adam`, `max_iter=500`, `early_stopping=True` | Jak Z4, Z6, Z7, Z8 |
| **RF 100 drzew** | Bez skalowania | Baseline z PDF i Z11 |
| **`cross_val_score`, cv=5** | Accuracy dla 4 modeli | Porównanie na train (przed finalnym testem) |
| **Test 80/20, `stratify=y`** | MLP `(32, 16)` vs RF | „Środkowa” architektura MLP — nie trzeba wybierać zwycięzcy z CV |
| **Bar chart** | 4 słupki = średnia CV accuracy | Szybkie porównanie wizualne |

**Parametry wyłącznie z polecenia:** próbka 800, 3 architektury, RF 100 drzew, parametry MLP z tabeli w notebooku.

---

### Co oznaczają typowe wyniki?

Typowe wyniki po uruchomieniu (mogą minimalnie się różnić):

| Model | CV accuracy (średnia) | Std | Czas CV | Interpretacja |
|-------|----------------------|-----|---------|---------------|
| **MLP (32,)** | ~0.81 | ~0.02 | ~0.1 s | 1 warstwa — za mało pojemności na 11 cech |
| **MLP (32, 16)** | ~0.86 | ~0.01 | ~0.1 s | Najlepszy MLP z trzech — 2 warstwy wystarczają |
| **MLP (64, 32)** | ~0.85 | ~0.004 | ~0.08 s | Więcej neuronów ≠ zawsze lepiej (overfitting / szum) |
| **RF (100 drzew)** | ~**0.88** | ~0.02 | ~0.5–0.7 s | **Wygrywa CV** — typowe na tabular |

#### Test accuracy (MLP (32, 16) vs RF)

| Model | Test accuracy | Czas treningu |
|-------|---------------|---------------|
| **MLP (32, 16)** | ~0.84 | ~0.03 s |
| **RF (100 drzew)** | ~**0.92** | ~0.1 s |

RF często **przewyższa MLP na teście** — to zgodne z PDF lekcji 24: na danych tabelarycznych ensemble drzew bywa silniejszy niż prosta sieć sklearn.

#### Dlaczego RF wygrywa?

- **11 cech, ~800 wierszy** — mały, klasyczny problem tabularny
- Drzewa dobrze łapią **nieliniowe progi** bez skalowania
- MLP w sklearn to **płytka sieć** — bez deep learningu, augmentacji, dużej ilości danych
- **Nierównowaga klas** (~14% „good”) — oba modele mogą mieć wysoką accuracy przez dominację klasy 0; warto patrzeć też na precision/recall w projekcie

#### Dlaczego MLP (64, 32) nie bije (32, 16)?

- Większa sieć = więcej parametrów → ryzyko **overfittingu** na małym zbiorze
- Reguła kciuka z PDF: zacznij od 16–64 neuronów; **więcej nie zawsze lepiej**

---

### Jak czytać wykres słupkowy CV?

- **4 słupki** — 3 MLP + RF; oś Y = accuracy
- **RF najwyżej** → na tym zbiorze ensemble jest lepszym wyborem
- **MLP (32,)** najniżej → jedna warstwa za słaba na 11 wymiarów
- Różnice MLP między sobą ~3–5 pp — wybór architektury ma znaczenie, ale RF i tak wygrywa

---

### Z11 vs Wine — ten sam motyw, inny kontekst

| | **Zadanie 11 (make_moons)** | **Wine Quality** |
|---|----------------------------|------------------|
| **Dane** | Syntetyczne 2D | Tabelaryczne 11 cech |
| **Cel** | Granice decyzyjne (`contourf`) | Metryki + bar chart CV |
| **Skalowanie MLP** | Ręczny `StandardScaler` | `Pipeline` |
| **Walidacja** | Jeden train/test | CV=5 + test |
| **Typowy wynik** | MLP ≈ RF (podobna accuracy) | **RF > MLP** |
| **Wniosek** | Oba uczą nieliniowości — widać na wykresie | Na tabular RF często baseline do pokonania |

---

### Czy będziesz tego używać w pracy DS?

**Wine Quality jako projekt — raczej nie** (to klasyczny dataset edukacyjny).

**Wzorzec zadania — tak, bardzo:**

| Element | Zastosowanie w pracy |
|---------|---------------------|
| **Porównanie MLP vs RF/XGBoost** | Baseline przy każdym nowym problemie tabularnym |
| **Pipeline + CV** | Standard przed wyborem modelu |
| **Zamknięta lista architektur** | Kontrolowany eksperyment — zmieniasz jedno na raz |
| **Bar chart CV** | Prezentacja wyników zespołowi |
| **Wniosek „RF ≥ MLP na tabular”** | Decyzja o algorytmie — oszczędza czas |

Sieci neuronowe w DS produkcyjnym: **obrazy, NLP, sekwencje**; tabular → **XGBoost, LightGBM, RF, regresja logistyczna**.

---

### Szablon odpowiedzi na rozmowę (30 s)

> „Na Wine Quality porównałam trzy architektury MLP z Random Forest — 800 próbek, target quality >= 7. Użyłam Pipeline ze skalerem i walidacji krzyżowej cv=5, tak jak w porównaniu aktywacji na Breast Cancer. Random Forest miał ~88% accuracy w CV i ~92% na teście, MLP (32, 16) ~86% i ~84%. To typowy wzorzec: na małych danych tabelarycznych ensemble drzew często wygrywa z prostą siecią sklearn. W projekcie zacząłbym od RF lub XGBoost, a MLP rozważył przy obrazach lub tekście.”

---

### Podsumowanie

| Element | Wniosek |
|---------|---------|
| **Wine Quality** | Tabular, binary, nierównowaga klas — realistyczny, ale edukacyjny |
| **800 próbek** | Kompromis szybkość vs reprezentatywność |
| **3 MLP + RF** | Kontrolowane porównanie — parametry z tabeli |
| **Pipeline + CV=5** | Obowiązkowy wzorzec przy porównywaniu modeli |
| **RF > MLP** | Typowy wynik na tabular — zgodnie z PDF |
| **Test MLP (32,16) vs RF** | Finalna ocena na danych niewidzianych |
| **Praktyka DS** | Ten sam workflow co w każdym projekcie ML |

---
