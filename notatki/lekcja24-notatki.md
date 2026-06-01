# Lekcja 24 — krótkie notatki
Skrócone notatki do szybkiego powtórzenia. Pełne wersje: [`lekcja24-powtorka.md`](lekcja24-powtorka.md).
---

## Zadanie 1

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


**Podsumowanie wyników:** im większe różnice między logitami, tym bardziej „ostry” rozkład (Test 3). Im bardziej zbliżone logity, tym bardziej „miękki” i niepewny rozkład (Test 1). Równe logity → brak preferencji (Test 2).

---


## Zadanie 4

### O czym jest to zadanie?

Trenujesz **gotową sieć neuronową** (`MLPClassifier` ze sklearn) na datasecie **Iris** — klasyfikacja 3 gatunków kwiatów na podstawie 4 cech (długość/szerokość płatków i działek).

To **nie** jest pisanie sieci od zera. To standardowy **pipeline ML**:

```
dane → podział train/test → skalowanie → model → predykcja → metryki → wykres loss
```

Architektura sieci: **4 wejścia → 10 neuronów → 5 neuronów → 3 klasy**.


## Zadanie 5

### O czym jest to zadanie?

Obliczasz **ile parametrów (wag + biasów)** ma sieć neuronowa MLP dla różnych architektur — bez trenowania modelu, czysto matematycznie.

To odpowiedź na pytanie: *„Jak duża jest ta sieć?”* — im więcej parametrów, tym bardziej **złożony** model (większa pojemność, ale też ryzyko overfittingu i dłuższy trening).


### Podsumowanie

| Aspekt | Ocena |
|--------|-------|
| Kod `count_parameters()` w pracy | Rzadko |
| Intuicja „rozmiar modelu” | **Bardzo przydatna** |
| Wzór wagi + bias | Warto umieć wytłumaczyć |
| Podobne zadanie na rozmowie | Raczej pytanie teoretyczne, nie coding challenge |
| Barplot architektur | Edukacyjnie — pokazuje skok złożoności |

**Zadanie uczy myślenia o złożoności modelu**, nie codziennego liczenia parametrów. To ważne przy doborze architektury, regularyzacji i rozmowach o overfittingu — nawet jeśli w projekcie używasz gotowych bibliotek.


## Zadanie 6

### O czym jest to zadanie?

Porównujesz **3 funkcje aktywacji** w tej samej sieci MLP na datasecie **Breast Cancer** — jedyna zmienna to `activation`, reszta identyczna (architektura, solver, liczba epok).

Pytanie badawcze: *„Która aktywacja daje najlepszą klasyfikację na tym zbiorze?”*

To eksperyment **kontrolowany** — zmieniasz jeden element i obserwujesz wpływ na wynik.


### Podsumowanie

| Element | Wniosek |
|---------|---------|
| **ReLU** | Najlepsza średnia accuracy — domyślny wybór w praktyce |
| **Logistic** | Prawie tak samo dobra, najstabilniejsza między foldami |
| **Tanh** | Nieco słabsza, ale nadal >96% |
| **CV vs jeden test** | CV daje wiarygodniejsze porównanie |
| **Pipeline** | Obowiązkowy przy skalowaniu + CV |
| **Praktyka DS** | Wzorzec „porównaj warianty przez CV” — uniwersalny; aktywacje MLP — niszowo |


### Na rozmowę

> „Porównałam 3 aktywacje w MLP na Breast Cancer z walidacją krzyżową cv=5. Użyłam Pipeline ze StandardScaler, żeby skalowanie było poprawne w każdym foldzie. ReLU dało najwyższą średnią accuracy (~97.7%), tanh nieco gorsze (~96.7%). Różnice były małe — na tym zbiorze wybór aktywacji nie ma dramatycznego wpływu. Ważniejsze było użycie CV zamiast jednego podziału train/test.”


## Zadanie 7

### O czym jest to zadanie?

Trenujesz **sieć neuronową do regresji** (`MLPRegressor`) — zamiast klasy (0/1/2) przewidujesz **liczbę ciągłą**: mediana ceny domu w California Housing.

To **Zadanie 4 w wersji regresyjnej** — ten sam typ modelu (MLP), inny typ problemu (regresja zamiast klasyfikacji).

```
cechy (dochód, wiek domu, współrzędne...) → MLP → przewidywana cena
```


### Podsumowanie

| Element | Wniosek |
|---------|---------|
| **MLPRegressor** | Sieć neuronowa do regresji — ten sam rodzina co MLPClassifier |
| **RMSE / MAE / R²** | **Must know** — standard oceny regresji |
| **Scatter pred vs true** | Szybka wizualna ocena — stosuj w każdym projekcie regresyjnym |
| **Early stopping** | Przydatna technika — oszczędza czas, ogranicza overfitting |
| **Próbka 3000** | Szybszy eksperyment — w produkcji trenujesz na pełnych danych |
| **Praktyka DS** | Metryki i wykresy — tak; MLPRegressor na tabular — raczej nie |


### Na rozmowę

> „Trenowałam MLPRegressor na próbce California Housing — regresja, przewidywanie mediana ceny. Użyłam architektury (32, 16) z early stopping. Na teście: RMSE ~0.59, R² ~0.73 — model wyjaśnia ok. 73% wariancji. Scatter pokazał, że predykcje są blisko linii idealnej, ale z rozrzutem. W praktyce na tabular data częściej używam XGBoost lub regresji liniowej, ale metryki RMSE/MAE/R² stosuję zawsze.”


## Zadanie 8

### O czym jest to zadanie?

Porównujesz **dwa sposoby zatrzymania treningu** tego samego MLP na Breast Cancer — jedyna różnica to `early_stopping=True` vs `False`.

Pytanie badawcze: *„Czy warto przerywać trening wcześniej — i co za to płacimy?”*

```
early_stopping=True  → stop, gdy model przestaje się poprawiać na walidacji
early_stopping=False → trenuj do max_iter (1000) lub pełnej zbieżności
```


### Podsumowanie

| Element | Wniosek |
|---------|---------|
| **Early stopping** | Przerwij trening, gdy model przestaje się poprawiać |
| **Czas vs epoki** | True = szybciej i mniej iteracji |
| **Accuracy** | Często podobna — nie zawsze warto trenować do końca |
| **Loss curve** | Wizualna diagnostyka — gdzie model się „nasycił” |
| **Praktyka DS** | Włączaj early stopping przy modelach iteracyjnych |
| **Kod od zera** | Nie — wystarczy znać parametr i intuicję |


### Na rozmowę

> „Porównałam MLP z early stopping włączonym i wyłączonym na Breast Cancer. Z early stopping trening trwał krócej i używał mniej epok, bo sklearn zatrzymuje model, gdy loss na walidacji przestaje spadać. Accuracy na teście była podobna — czasem nieco niższa przy True, ale oszczędność czasu jest duża. To forma regularyzacji — chroni przed overfittingiem. Ten sam pomysł stosuję w XGBoost przez early_stopping_rounds.”


## Zadanie 11

### O czym jest to zadanie?

Porównujesz **MLP** i **Random Forest** na syntetycznych danych `make_moons` — problem **nieliniowo rozdzielny** (jedna linia go nie rozwiąże). Wizualizujesz **granice decyzyjne** obu modeli obok siebie.

```
make_moons(500, noise=0.3) → MLP (20, 10) vs RF (100 drzew) → contourf + accuracy, czas
```

### Podsumowanie

| Element | Wniosek |
|---------|---------|
| **make_moons** | Klasyczny test nieliniowości — „księżyce” nie da się rozdzielić linią |
| **MLP (20, 10)** | Gładka, zakrzywiona granica — sieć uczy się nieliniowości |
| **RF (100 drzew)** | Poszarpana granica — każde drzewo tnie przestrzeń prostopadle |
| **Accuracy** | Często podobna (~0.87–0.89) — inny kształt granicy, podobny wynik |
| **Czas** | Oba modele szybkie na 500 próbkach; RF często równie szybki lub szybszy |
| **Skalowanie MLP** | Ręczny `StandardScaler` (fit na train) — **bez Pipeline**, jak PDF sekcja 6.1 |
| **Pipeline** | Nie jest wymagany — jeden podział train/test + wykres; Pipeline potrzebny przy CV (Z6, Wine) |
| **Praktyka DS** | RF/XGBoost częstsze na tabular; MLP/deep learning — obrazy, tekst; granice decyzyjne — do wizualizacji i nauki |

### Na rozmowę

> „Na make_moons porównałam MLP (20, 10) z Random Forest (100 drzew). Oba osiągnęły podobną accuracy (~87–92%), ale granice decyzyjne wyglądają inaczej — MLP ma gładką krzywą, RF poszarpaną. MLP trenowałam na skalowanych danych, RF na surowych — bez Pipeline, bo to jeden podział i wizualizacja, nie CV. make_moons to problem nieliniowy — perceptron by tu nie zadziałał.”


## Zadanie dodatkowe — Wine Quality

### O czym jest to zadanie?

Porównujesz **3 gotowe architektury MLP** z **Random Forest** na **danych tabelarycznych** Wine Quality (czerwone wino, UCI). Target: `quality >= 7` → klasa „good” (1). To **kontynuacja lekcji 23** — ten sam dataset, inne modele (sieć vs ensemble).

```
800 próbek → CV=5 (4 modele) → bar chart → test: MLP (32, 16) vs RF
```

Parametry **zamknięte w tabeli zadania** — nie wymyślasz architektur ani hiperparametrów.

### Podsumowanie

| Element | Wniosek |
|---------|---------|
| **800 próbek** | Szybsze CV niż pełne ~1600 wierszy; nadal mały dataset tabularny |
| **3 MLP + RF** | `(32,)`, `(32, 16)`, `(64, 32)` vs 100 drzew — stała lista do porównania |
| **Pipeline + CV** | `StandardScaler` w Pipeline + `cross_val_score(cv=5)` — wzorzec z Zadania 6 |
| **Typowe CV** | RF ~88% > MLP (32,16) ~86% > MLP (64,32) ~85% > MLP (32,) ~81% |
| **Typowy test** | RF często wygrywa (~90–92%) vs MLP (32,16) (~84%) |
| **Nierównowaga klas** | ~14% „good” — stąd `stratify=y` przy podziale |
| **Praktyka DS** | Na tabular data **RF/XGBoost** często ≥ MLP — zgodnie z PDF lekcji 24 |

### Na rozmowę

> „Na Wine Quality porównałam trzy architektury MLP z Random Forest. Użyłam Pipeline ze skalerem i walidacji krzyżowej cv=5 — tak jak przy porównywaniu aktywacji w Zadaniu 6. Random Forest miał wyższą accuracy (~88% CV, ~92% test) niż MLP (32, 16). To typowy wynik na małych danych tabelarycznych: ensemble drzew często wygrywa z prostą siecią sklearn. Sieci neuronowe stosuję raczej przy obrazach i tekście.”

