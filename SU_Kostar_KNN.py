import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from collections import Counter

def get_colors(n_classes):
    """Vygeneruje n rôznych farieb - zelená, modrá, červená,..."""
    cmap = plt.get_cmap('tab10')
    # Poradie: zelená(2), modrá(0), červená(3), oranžová(1), fialová(4),...
    color_order = [2, 0, 3, 1, 4, 5, 6, 7, 8, 9]
    return [cmap(color_order[i % len(color_order)]) for i in range(n_classes)]


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_test):
        X_test = np.array(X_test)
        dists_sq = (np.sum(X_test ** 2, axis=1)[:, np.newaxis]
                    + np.sum(self.X_train ** 2, axis=1)
                    - 2 * np.dot(X_test, self.X_train.T))
        dists = np.sqrt(np.maximum(dists_sq, 0))
        k_indices = np.argsort(dists, axis=1)[:, :self.k]
        k_labels = self.y_train[k_indices]
        return np.array([Counter(row).most_common(1)[0][0] for row in k_labels])


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("KNN Zadanie")
        self.root.geometry("1500x900")

        self.df = None
        self.class_names = []

        # --- BOČNÝ PANEL ---
        controls = ttk.Frame(root, padding="15")
        controls.pack(side=tk.LEFT, fill=tk.Y)

        # 1. Dáta
        ttk.Label(controls, text="Dáta", font=('Arial', 10, 'bold')).pack(pady=5)
        self.dataset_var = tk.StringVar(value="Iris")
        self.dataset_combo = ttk.Combobox(controls, textvariable=self.dataset_var, state="readonly")
        self.dataset_combo['values'] = ("Iris", "Wine", "Breast Cancer", "Digits", "Vlastné CSV...")
        self.dataset_combo.pack(fill=tk.X)
        self.dataset_combo.bind("<<ComboboxSelected>>", self.handle_dataset_change)
        ttk.Button(controls, text="Nahrať CSV", command=self.load_csv).pack(fill=tk.X, pady=5)

        # 2. Režim zobrazenia
        ttk.Label(controls, text="Zobrazenie", font=('Arial', 10, 'bold')).pack(pady=10)
        self.mode_var = tk.StringVar(value="manual")
        ttk.Radiobutton(controls, text="Manuálny výber (2 osi)", variable=self.mode_var, value="manual").pack(
            anchor=tk.W)
        ttk.Radiobutton(controls, text="PCA (Všetky priznaky -> 2D)", variable=self.mode_var, value="pca").pack(
            anchor=tk.W)

        self.hide_train_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Skryť trénovacie body", variable=self.hide_train_var).pack(anchor=tk.W, pady=5)

        # 3. Výber stĺpcov (aktívne len v manuálnom režime)
        self.col_frame = ttk.Frame(controls)
        self.col_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.col_frame, text="Os X:").pack(anchor=tk.W)
        self.cb_x = ttk.Combobox(self.col_frame, state="readonly");
        self.cb_x.pack(fill=tk.X)
        ttk.Label(self.col_frame, text="Os Y:").pack(anchor=tk.W)
        self.cb_y = ttk.Combobox(self.col_frame, state="readonly");
        self.cb_y.pack(fill=tk.X)

        ttk.Label(controls, text="Cieľ (Label):").pack(anchor=tk.W, pady=(5, 0))
        self.cb_target = ttk.Combobox(controls, state="readonly");
        self.cb_target.pack(fill=tk.X)
        self.cb_target.bind("<<ComboboxSelected>>", lambda e: self.update_class_legend())

        # 3. Nastavenia splitu a K
        ttk.Label(controls, text="Parametre", font=('Arial', 10, 'bold')).pack(pady=10)

        self.k_label = ttk.Label(controls, text="K-susedia: 3")
        self.k_label.pack()
        self.k_slider = ttk.Scale(controls, from_=1, to=15, orient=tk.HORIZONTAL,
                                  command=lambda v: self.k_label.config(text=f"K-susedia: {int(float(v))}"))
        self.k_slider.set(3);
        self.k_slider.pack(fill=tk.X, pady=5)

        self.split_label = ttk.Label(controls, text="Trénovacia sada: 70%")
        self.split_label.pack()
        self.split_slider = ttk.Scale(controls, from_=10, to=90, orient=tk.HORIZONTAL,
                                      command=lambda v: self.split_label.config(
                                          text=f"Trénovacia sada: {int(float(v))}%"))
        self.split_slider.set(70);
        self.split_slider.pack(fill=tk.X, pady=5)

        ttk.Button(controls, text="AKTUALIZOVAŤ GRAF", command=self.update_plot).pack(fill=tk.X, pady=20)

        # 4. Legenda tried
        ttk.Label(controls, text="4. TRIEDY (LEGENDA)", font=('Arial', 10, 'bold')).pack(pady=10)
        self.class_frame = ttk.Frame(controls)
        self.class_frame.pack(fill=tk.X, pady=5)

        # Metriky a Graf
        self.stats_text = tk.StringVar(value="Pripravený...")
        ttk.Label(root, textvariable=self.stats_text, font=('Consolas', 11)).pack(side=tk.BOTTOM, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(7, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.load_builtin_data()

    def load_builtin_data(self):
        name = self.dataset_var.get()
        if name == "Iris":
            d = datasets.load_iris()
            self.df = pd.DataFrame(d.data, columns=d.feature_names)
            self.df['species'] = d.target
            self.class_names = d.target_names
        elif name == "Wine":
            d = datasets.load_wine()
            self.df = pd.DataFrame(d.data, columns=d.feature_names)
            self.df['class'] = d.target
            self.class_names = d.target_names
        elif name == "Breast Cancer":
            d = datasets.load_breast_cancer()
            self.df = pd.DataFrame(d.data, columns=d.feature_names)
            self.df['target'] = d.target
            self.class_names = d.target_names
        elif name == "Digits":
            d = datasets.load_digits()
            self.df = pd.DataFrame(d.data)
            self.df['digit'] = d.target
            self.class_names = [str(i) for i in range(10)]
        self.refresh_column_options()

    def load_csv(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if fp:
            self.df = pd.read_csv(fp);
            self.dataset_var.set("Vlastné CSV...");
            self.refresh_column_options()

    def handle_dataset_change(self, event):
        if self.dataset_var.get() != "Vlastné CSV...": self.load_builtin_data()

    def refresh_column_options(self):
        if self.df is not None:
            cols = [c for c in self.df.columns if self.df[c].dtype in [np.float64, np.int64]]
            self.cb_x['values'] = cols;
            self.cb_y['values'] = cols;
            self.cb_target['values'] = list(self.df.columns)
            self.cb_x.current(0);
            self.cb_y.current(1);
            self.cb_target.current(len(self.df.columns) - 1)
            self.update_class_legend()

    def update_class_legend(self):
        # Vyčisti starú legendu
        for widget in self.class_frame.winfo_children():
            widget.destroy()

        if self.df is None:
            return

        target_col = self.cb_target.get()
        if not target_col or target_col not in self.df.columns:
            return

        # Získaj unikátne triedy (sorted pre konzistentné poradie)
        unique_vals = sorted(self.df[target_col].unique())

        # Pre built-in datasety mapuj číselné hodnoty na class_names
        if self.dataset_var.get() != "Vlastné CSV..." and len(self.class_names) > 0:
            labels = [self.class_names[v] if isinstance(v, (int, np.integer)) and v < len(self.class_names) else str(v) for v in unique_vals]
        else:
            labels = [str(v) for v in unique_vals]

        # Vytvor legendu s farbami
        colors = get_colors(len(labels))
        for i, label in enumerate(labels):
            row = ttk.Frame(self.class_frame)
            row.pack(fill=tk.X, pady=2)

            # Farebný štvorec
            color = '#%02x%02x%02x' % tuple(int(c * 255) for c in colors[i][:3])
            canvas = tk.Canvas(row, width=15, height=15, bg=color, highlightthickness=1,
                              highlightbackground="black")
            canvas.pack(side=tk.LEFT, padx=2)

            # Názov triedy
            ttk.Label(row, text=f"  {label}").pack(side=tk.LEFT)

    def update_plot(self):
        if self.df is None: return
        try:
            target_col = self.cb_target.get()
            k = int(self.k_slider.get())
            train_size = self.split_slider.get() / 100.0

            # Príprava dát podľa režimu
            if self.mode_var.get() == "pca":
                # Vezmeme všetky číselné stĺpce okrem targetu
                features = [c for c in self.df.columns if
                            c != target_col and self.df[c].dtype in [np.float64, np.int64]]
                X_full = self.df[features].values
                # PCA transformácia na 2 komponenty
                pca = PCA(n_components=2)
                X = pca.fit_transform(X_full)
                label_x, label_y = "PCA komponent 1", "PCA komponent 2"
            else:
                X = self.df[[self.cb_x.get(), self.cb_y.get()]].values
                label_x, label_y = self.cb_x.get(), self.cb_y.get()

            y_raw = self.df[target_col].values
            y, uniques = pd.factorize(y_raw)
            # Pre built-in datasety mapuj číselné hodnoty na class_names
            if self.dataset_var.get() != "Vlastné CSV..." and len(self.class_names) > 0:
                current_labels = [self.class_names[i] if isinstance(i, (int, np.integer)) and i < len(self.class_names) else str(i) for i in uniques]
            else:
                current_labels = [str(u) for u in uniques]

            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

            clf = KNN(k=k);
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            correct = np.sum(preds == y_test)
            incorrect = len(y_test) - correct
            acc = (correct / len(y_test)) * 100
            self.stats_text.set(f"Režim: {self.mode_var.get().upper()} | Presnosť: {acc:.1f}% | Správne: {correct}/{len(y_test)} | Nesprávne: {incorrect} | K={k}")

            # VYKRESLENIE
            self.ax.clear()
            colors = get_colors(len(current_labels))
            cmap_custom = plt.cm.colors.ListedColormap(colors)

            # Hranice
            h = (X[:, 0].max() - X[:, 0].min()) / 100
            xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, h),
                                 np.arange(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            self.ax.contourf(xx, yy, Z, alpha=0.15, cmap=cmap_custom)

            for i, name in enumerate(current_labels):
                # Trénovacie body (skryť ak je checkbox zakliknutý)
                if not self.hide_train_var.get():
                    idx_tr = np.where(y_train == i)
                    self.ax.scatter(X_train[idx_tr, 0], X_train[idx_tr, 1], color=colors[i],
                                    marker='x', s=60, label=f"{name} (Train)")
                # Testovacie body (vždy zobraziť)
                idx_ts = np.where(y_test == i)
                self.ax.scatter(X_test[idx_ts, 0], X_test[idx_ts, 1], color=colors[i],
                                marker='o', s=80, edgecolors='k', alpha=0.8, label=f"{name} (Test)")

            self.ax.set_xlabel(label_x);
            self.ax.set_ylabel(label_y)
            self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small')
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Chyba", str(e))


if __name__ == "__main__":
    root = tk.Tk();
    app = App(root);
    root.mainloop()