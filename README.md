# Interaktívna Vizualizácia KNN Algoritmu

Tento projekt je desktopová aplikácia napísaná v jazyku Python, ktorá slúži na interaktívnu vizualizáciu klasifikačného algoritmu **K-Nearest Neighbors (K-najbližších susedov)**. Aplikácia umožňuje používateľovi v reálnom čase sledovať, ako sa menia rozhodovacie hranice modelu pri zmene rôznych parametrov.

##Funkcie

- **Podpora viacerých datasetov:** Integrované datasety (Iris, Wine, Breast Cancer, Digits) s možnosťou nahrať vlastný súbor vo formáte `.csv`.
- **Dva režimy zobrazenia:**
  - *Manuálny výber:* Používateľ si zvolí konkrétne stĺpce pre os X a os Y.
  - *PCA (Analýza hlavných komponentov):* Redukcia viacerých dimenzií do 2D priestoru pre komplexné datasety.
- **Interaktívne ovládanie:**
  - Slider pre nastavenie počtu susedov (**K**).
  - Slider pre nastavenie pomeru trénovacej a testovacej množiny (**Split**).
- **Dynamická legenda:** Farby a názvy tried sa automaticky prispôsobujú zvolenému datasetu.
- **Štatistiky v reálnom čase:** Zobrazenie presnosti (Accuracy) a počtu správne/nesprávne klasifikovaných bodov.

## Požiadavky

Pre spustenie aplikácie je potrebné mať nainštalovaný Python 3.x a nasledujúce knižnice:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Tieto knižnice nainštalujete príkazom:

```bash
pip install numpy pandas matplotlib scikit-learn
