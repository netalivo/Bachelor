# Benchmark zur Residualanalyse zur Bewertung des Prognoseerfolgs

Dies ist das Quellcode-Repository zur Bachelorarbeit **„Benchmark zur Residualanalyse zur Bewertung des Prognoseerfolgs“**.

## Wichtige Notebooks & Dateien
### Jupyter Notebooks
* `main_single_store_additive.ipynb` & `main_single_store_sarima.ipynb` – visualisieren die Zeitreihe und Prognosen **einzelner Filialen** sowie die p-Werte oder Statistiken der statistischen Tests für die Naiv-Methode mit saisonalem Drift (Naiv-Modell) und das SARIMA-Modell, jeweils In-Sample und Out-of-Sample.  
* `main_all_stores_additive.ipynb` & `main_all_stores_SARIMA.ipynb` – erzeugen Heatmaps, Boxplots und Scatterplots der p-Werte oder Statistiken der statistischen Tests **aller Filialen** analog zu Kapitel 4 der Bachelorarbeit (Naiv-Modell und SARIMA-Modell, In- & Out-of-Sample).
* `test.ipynb` – zusätzliche Grafiken, die in Kapitel 4 abgebildet sind.
### Python Dateien
* `model_naiv.py` – Implementierung der Naiv-Methode mit saisonalem Drift (Naiv-Modell) (und der einfachen saisonalen Naiv-Methode).
* `model_SARIMA.py` – Implementierung des SARIMA-Modells. Enthält außerdem sämtliche automatisch berechneten Orders.  
* `test_correlation.py` – Autokorrelationtests.  
* `test_mean.py` – Mittelwerttests.
### Datensatz
* `Walmart_Sales.csv` – enthält den im Benchmark verwendeten Datensatz.

## Nicht Teil der Arbeit aber möglicherweise interessant
* `dataset.py` & `main_dataset.ipynb` – berechnen und visualisieren die Beziehungen zwischen den externen Einflussfaktoren des Datensatzes.
* `main_all_stores_naiv.ipynb` – erzeugt Heatmaps, Boxplots und Scatterplots der p-Werte oder Statistiken der statistischen Tests aller Filialen für die einfache saisonale Naiv-Methode.
* `cross_validation.py` – Implementierung der Kreuzvalidierung.
