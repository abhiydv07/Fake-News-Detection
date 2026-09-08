"""
Tkinter GUI Module
Provides a graphical interface for the Fake News Detection system.
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from models import ModelTrainer, MODELS
from evaluation import ModelEvaluator
from sklearn.model_selection import train_test_split


class FakeNewsDetectorApp:
    """Main application window."""

    def __init__(self, root):
        self.root = root
        self.root.title("Fake News Detection Using Machine Learning")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)

        # Set style
        style = ttk.Style()
        style.theme_use('clam')

        # State variables
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trainer = None
        self.evaluator = ModelEvaluator()
        self.feature_extractor = None
        self.preprocessor = None
        self.all_results = {}

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Create tabs
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_eval = ttk.Frame(self.notebook)
        self.tab_predict = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_data, text="  📊 Data  ")
        self.notebook.add(self.tab_train, text="  🤖 Training  ")
        self.notebook.add(self.tab_eval, text="  📈 Evaluation  ")
        self.notebook.add(self.tab_predict, text="  🔍 Predict  ")

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var,
                                     relief='sunken', anchor='w')
        self.status_bar.pack(fill='x', side='bottom')

        self._setup_data_tab()
        self._setup_train_tab()
        self._setup_eval_tab()
        self._setup_predict_tab()

    # ==================== DATA TAB ====================
    def _setup_data_tab(self):
        """Setup the data loading and preview tab."""
        # Top frame for controls
        ctrl_frame = ttk.LabelFrame(self.tab_data, text="Load Dataset", padding=10)
        ctrl_frame.pack(fill='x', padx=10, pady=5)

        # File path
        ttk.Label(ctrl_frame, text="Dataset Path:").grid(row=0, column=0, sticky='w')
        self.file_path_var = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self.file_path_var, width=60).grid(
            row=0, column=1, padx=5, sticky='ew')
        ttk.Button(ctrl_frame, text="Browse", command=self._browse_file).grid(row=0, column=2)

        # Column settings
        ttk.Label(ctrl_frame, text="Text Column:").grid(row=1, column=0, sticky='w', pady=5)
        self.text_col_var = tk.StringVar(value='text')
        ttk.Entry(ctrl_frame, textvariable=self.text_col_var, width=20).grid(
            row=1, column=1, sticky='w', padx=5)

        ttk.Label(ctrl_frame, text="Title Column (optional):").grid(row=2, column=0, sticky='w', pady=5)
        self.title_col_var = tk.StringVar(value='title')
        ttk.Entry(ctrl_frame, textvariable=self.title_col_var, width=20).grid(
            row=2, column=1, sticky='w', padx=5)

        ttk.Label(ctrl_frame, text="Label Column:").grid(row=3, column=0, sticky='w', pady=5)
        self.label_col_var = tk.StringVar(value='label')
        ttk.Entry(ctrl_frame, textvariable=self.label_col_var, width=20).grid(
            row=3, column=1, sticky='w', padx=5)

        ttk.Button(ctrl_frame, text="Load & Preview", command=self._load_data).grid(
            row=4, column=1, pady=10, sticky='w')

        ctrl_frame.columnconfigure(1, weight=1)

        # Data preview
        preview_frame = ttk.LabelFrame(self.tab_data, text="Data Preview", padding=5)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Stats labels
        self.data_stats_var = tk.StringVar(value="No data loaded.")
        ttk.Label(preview_frame, textvariable=self.data_stats_var,
                  font=('Segoe UI', 10, 'bold')).pack(anchor='w')

        # Treeview for data preview
        tree_frame = ttk.Frame(preview_frame)
        tree_frame.pack(fill='both', expand=True)

        self.tree = ttk.Treeview(tree_frame, show='headings', height=15)
        scrollbar_y = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(tree_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        self.tree.grid(row=0, column=0, sticky='nsew')
        scrollbar_y.grid(row=0, column=1, sticky='ns')
        scrollbar_x.grid(row=1, column=0, sticky='ew')

        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

    def _browse_file(self):
        """Open file dialog to select dataset."""
        filepath = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.file_path_var.set(filepath)

    def _load_data(self):
        """Load and preview the selected dataset."""
        filepath = self.file_path_var.get().strip()
        if not filepath:
            messagebox.showerror("Error", "Please select a dataset file.")
            return

        try:
            # Load data
            if filepath.endswith('.csv'):
                self.df = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(filepath)
            else:
                messagebox.showerror("Error", "Unsupported file format.")
                return

            # Update stats
            label_col = self.label_col_var.get()
            if label_col in self.df.columns:
                fake_count = (self.df[label_col] == 1).sum()
                real_count = (self.df[label_col] == 0).sum()
                stats = f"Total: {len(self.df)} articles | Real: {real_count} | Fake: {fake_count}"
            else:
                stats = f"Total: {len(self.df)} rows | Columns: {list(self.df.columns)}"

            self.data_stats_var.set(stats)

            # Update treeview
            self.tree.delete(*self.tree.get_children())
            self.tree['columns'] = list(self.df.columns[:5])

            for col in self.df.columns[:5]:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=200)

            for idx, row in self.df.head(100).iterrows():
                values = [str(v)[:80] for v in row[:5]]
                self.tree.insert('', 'end', values=values)

            self.status_var.set(f"Loaded {len(self.df)} rows from {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")

    # ==================== TRAINING TAB ====================
    def _setup_train_tab(self):
        """Setup the model training tab."""
        ctrl_frame = ttk.LabelFrame(self.tab_train, text="Training Configuration", padding=10)
        ctrl_frame.pack(fill='x', padx=10, pady=5)

        # Preprocessing options
        ttk.Label(ctrl_frame, text="Preprocessing:", font=('Segoe UI', 10, 'bold')).grid(
            row=0, column=0, sticky='w', columnspan=2)

        self.use_stemming_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="Stemming", variable=self.use_stemming_var).grid(
            row=1, column=0, sticky='w')

        self.use_lemma_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="Lemmatization", variable=self.use_lemma_var).grid(
            row=1, column=1, sticky='w')

        self.remove_stop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="Remove Stop Words", variable=self.remove_stop_var).grid(
            row=2, column=0, sticky='w')

        # Feature extraction
        ttk.Label(ctrl_frame, text="Feature Extraction:", font=('Segoe UI', 10, 'bold')).grid(
            row=3, column=0, sticky='w', columnspan=2, pady=(10, 0))

        ttk.Label(ctrl_frame, text="Method:").grid(row=4, column=0, sticky='w')
        self.feature_method_var = tk.StringVar(value='tfidf')
        ttk.Combobox(ctrl_frame, textvariable=self.feature_method_var,
                      values=['tfidf', 'count'], width=15, state='readonly').grid(
            row=4, column=1, sticky='w', padx=5)

        ttk.Label(ctrl_frame, text="Max Features:").grid(row=5, column=0, sticky='w')
        self.max_features_var = tk.StringVar(value='5000')
        ttk.Entry(ctrl_frame, textvariable=self.max_features_var, width=10).grid(
            row=5, column=1, sticky='w', padx=5)

        ttk.Label(ctrl_frame, text="Test Size:").grid(row=6, column=0, sticky='w')
        self.test_size_var = tk.StringVar(value='0.2')
        ttk.Entry(ctrl_frame, textvariable=self.test_size_var, width=10).grid(
            row=6, column=1, sticky='w', padx=5)

        # Model selection
        ttk.Label(ctrl_frame, text="Models:", font=('Segoe UI', 10, 'bold')).grid(
            row=7, column=0, sticky='w', columnspan=2, pady=(10, 0))

        self.model_vars = {}
        model_names = list(MODELS.keys())
        for i, name in enumerate(model_names):
            var = tk.BooleanVar(value=True)
            self.model_vars[name] = var
            row = 8 + i // 3
            col = i % 3
            ttk.Checkbutton(ctrl_frame, text=name, variable=var).grid(
                row=row, column=col, sticky='w', padx=5)

        # Train button
        btn_frame = ttk.Frame(self.tab_train)
        btn_frame.pack(fill='x', padx=10, pady=5)

        self.train_btn = ttk.Button(btn_frame, text="🚀 Train All Models",
                                     command=self._start_training)
        self.train_btn.pack(side='left', padx=5)

        # Progress
        self.train_progress = ttk.Progressbar(self.tab_train, mode='indeterminate')
        self.train_progress.pack(fill='x', padx=10, pady=2)

        # Results text area
        result_frame = ttk.LabelFrame(self.tab_train, text="Training Results", padding=5)
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.train_output = scrolledtext.ScrolledText(result_frame, height=12,
                                                       font=('Consolas', 9))
        self.train_output.pack(fill='both', expand=True)

    def _start_training(self):
        """Start training in a background thread."""
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first (Data tab).")
            self.notebook.select(self.tab_data)
            return

        selected_models = [name for name, var in self.model_vars.items() if var.get()]
        if not selected_models:
            messagebox.showerror("Error", "Please select at least one model.")
            return

        self.train_btn.config(state='disabled')
        self.train_progress.start(10)
        self.train_output.delete('1.0', 'end')

        thread = threading.Thread(target=self._train_models, args=(selected_models,))
        thread.daemon = True
        thread.start()

    def _train_models(self, selected_models):
        """Train models (runs in background thread)."""
        try:
            text_col = self.text_col_var.get()
            title_col = self.title_col_var.get()
            label_col = self.label_col_var.get()
            test_size = float(self.test_size_var.get())
            max_features = int(self.max_features_var.get())

            self._log_train("=== Fake News Detection Pipeline ===\n")

            # Step 1: Preprocessing
            self._log_train("\n[1/4] Preprocessing text data...")
            self.preprocessor = TextPreprocessor(
                use_stemming=self.use_stemming_var.get(),
                use_lemmatization=self.use_lemma_var.get(),
                remove_stopwords=self.remove_stop_var.get()
            )
            df = self.preprocessor.preprocess_dataframe(
                self.df, text_column=text_col,
                title_column=title_col if title_col else None
            )
            self._log_train(f"  Preprocessed {len(df)} articles.")

            # Step 2: Feature Extraction
            self._log_train("\n[2/4] Extracting features...")
            self.feature_extractor = FeatureExtractor(
                method=self.feature_method_var.get(),
                max_features=max_features
            )
            X = self.feature_extractor.fit_transform(df['processed_text'])
            y = df[label_col].values
            self._log_train(f"  Feature matrix shape: {X.shape}")
            self._log_train(f"  Vocabulary size: {len(self.feature_extractor.get_feature_names())}")

            # Step 3: Split data
            self._log_train(f"\n[3/4] Splitting data (test_size={test_size})...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            self._log_train(f"  Training set: {self.X_train.shape[0]} samples")
            self._log_train(f"  Test set: {self.X_test.shape[0]} samples")

            # Step 4: Train models
            self._log_train(f"\n[4/4] Training {len(selected_models)} models...\n")
            self.trainer = ModelTrainer(selected_models)
            self.trainer.train_all(self.X_train, self.y_train)

            # Evaluate all models
            self.all_results = {}
            self._log_train("\n=== Evaluation Results ===\n")

            for name in selected_models:
                y_pred = self.trainer.predict(self.X_test, name)
                try:
                    y_prob = self.trainer.predict_proba(self.X_test, name)
                except Exception:
                    y_prob = None

                results = self.evaluator.evaluate(
                    self.y_test, y_pred, y_prob, model_name=name
                )
                self.all_results[name] = results

                self._log_train(f"--- {name} ---")
                self._log_train(f"  Accuracy:  {results['accuracy']:.4f}")
                self._log_train(f"  Precision: {results['precision']:.4f}")
                self._log_train(f"  Recall:    {results['recall']:.4f}")
                self._log_train(f"  F1 Score:  {results['f1_score']:.4f}")
                if results.get('roc_auc'):
                    self._log_train(f"  AUC:       {results['roc_auc']:.4f}")
                self._log_train("")

            # Find best model
            best_name = max(self.all_results,
                          key=lambda k: self.all_results[k]['accuracy'])
            best_acc = self.all_results[best_name]['accuracy']
            self._log_train(f"\n🏆 Best Model: {best_name} (Accuracy: {best_acc:.4f})")

            self.root.after(0, lambda: self.status_var.set(
                f"Training complete. Best: {best_name} ({best_acc:.2%})"))

        except Exception as e:
            self._log_train(f"\n❌ ERROR: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
        finally:
            self.root.after(0, self._training_done)

    def _log_train(self, msg):
        """Thread-safe log to training output."""
        self.root.after(0, lambda: (
            self.train_output.insert('end', msg + '\n'),
            self.train_output.see('end')
        ))

    def _training_done(self):
        """Clean up after training."""
        self.train_progress.stop()
        self.train_btn.config(state='normal')

    # ==================== EVALUATION TAB ====================
    def _setup_eval_tab(self):
        """Setup the evaluation and visualization tab."""
        ctrl_frame = ttk.Frame(self.tab_eval)
        ctrl_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(ctrl_frame, text="📊 Show Confusion Matrices",
                   command=self._show_confusion_matrices).pack(side='left', padx=5)
        ttk.Button(ctrl_frame, text="📈 Show ROC Curves",
                   command=self._show_roc_curves).pack(side='left', padx=5)
        ttk.Button(ctrl_frame, text="📉 Show Accuracy Comparison",
                   command=self._show_accuracy_comparison).pack(side='left', padx=5)
        ttk.Button(ctrl_frame, text="📋 Show Summary Table",
                   command=self._show_summary).pack(side='left', padx=5)

        # Chart area
        self.eval_frame = ttk.Frame(self.tab_eval)
        self.eval_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.chart_canvas = None

    def _clear_eval_frame(self):
        """Clear the evaluation display area."""
        for widget in self.eval_frame.winfo_children():
            widget.destroy()
        if self.chart_canvas:
            self.chart_canvas = None

    def _check_results(self):
        """Check if training results exist."""
        if not self.all_results:
            messagebox.showwarning("No Results", "Please train models first (Training tab).")
            self.notebook.select(self.tab_train)
            return False
        return True

    def _show_confusion_matrices(self):
        """Display confusion matrices for all models."""
        if not self._check_results():
            return

        self._clear_eval_frame()

        n = len(self.all_results)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig = Figure(figsize=(5 * cols, 4.5 * rows), dpi=100)
        for i, (name, results) in enumerate(self.all_results.items()):
            ax = fig.add_subplot(rows, cols, i + 1)
            self.evaluator.plot_confusion_matrix(
                results['confusion_matrix'], model_name=name, ax=ax
            )

        fig.tight_layout()
        self._display_figure(fig)

    def _show_roc_curves(self):
        """Display ROC curves for all models."""
        if not self._check_results():
            return

        self._clear_eval_frame()
        fig = Figure(figsize=(9, 6), dpi=100)
        ax = fig.add_subplot(111)
        self.evaluator.plot_roc_curve(self.all_results, ax=ax)
        self._display_figure(fig)

    def _show_accuracy_comparison(self):
        """Display accuracy comparison bar chart."""
        if not self._check_results():
            return

        self._clear_eval_frame()
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        self.evaluator.plot_accuracy_comparison(self.all_results, ax=ax)
        self._display_figure(fig)

    def _show_summary(self):
        """Display summary table."""
        if not self._check_results():
            return

        self._clear_eval_frame()

        summary = self.evaluator.get_summary_table(self.all_results)

        # Create table
        tree = ttk.Treeview(self.eval_frame,
                            columns=('Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'),
                            show='headings', height=len(summary))

        for col in ('Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'):
            tree.heading(col, text=col)
            tree.column(col, width=140, anchor='center')

        for row in summary:
            tree.insert('', 'end', values=[row[col] for col in
                        ('Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC')])

        tree.pack(fill='both', expand=True, padx=10, pady=10)

        # Highlight best model
        best = max(summary, key=lambda r: float(r['Accuracy']))
        self.status_var.set(f"Best Model: {best['Model']} (Accuracy: {best['Accuracy']})")

    def _display_figure(self, fig):
        """Display a matplotlib figure in the eval frame."""
        canvas = FigureCanvasTkAgg(fig, master=self.eval_frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self.eval_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.chart_canvas = canvas

    # ==================== PREDICT TAB ====================
    def _setup_predict_tab(self):
        """Setup the prediction tab."""
        top_frame = ttk.Frame(self.tab_predict)
        top_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(top_frame, text="Enter news text to classify:",
                  font=('Segoe UI', 11, 'bold')).pack(anchor='w')

        self.predict_text = scrolledtext.ScrolledText(top_frame, height=6,
                                                       font=('Segoe UI', 10))
        self.predict_text.pack(fill='x', pady=5)

        ctrl_frame = ttk.Frame(top_frame)
        ctrl_frame.pack(fill='x')

        ttk.Label(ctrl_frame, text="Select Model:").pack(side='left')
        self.predict_model_var = tk.StringVar(value='Logistic Regression')
        self.predict_model_combo = ttk.Combobox(
            ctrl_frame, textvariable=self.predict_model_var,
            values=list(MODELS.keys()), state='readonly', width=25
        )
        self.predict_model_combo.pack(side='left', padx=5)

        ttk.Button(ctrl_frame, text="🔍 Analyze",
                   command=self._predict_news).pack(side='left', padx=10)

        ttk.Button(ctrl_frame, text="Clear",
                   command=self._clear_prediction).pack(side='left')

        # Result display
        result_frame = ttk.LabelFrame(self.tab_predict, text="Prediction Result", padding=10)
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.result_label = ttk.Label(result_frame, text="Enter text and click Analyze",
                                       font=('Segoe UI', 14))
        self.result_label.pack(pady=10)

        self.result_detail = scrolledtext.ScrolledText(result_frame, height=10,
                                                        font=('Consolas', 9))
        self.result_detail.pack(fill='both', expand=True)

    def _predict_news(self):
        """Predict whether a news article is real or fake."""
        text = self.predict_text.get('1.0', 'end').strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze.")
            return

        if self.trainer is None or self.preprocessor is None:
            messagebox.showerror("Error", "Please train models first (Training tab).")
            self.notebook.select(self.tab_train)
            return

        model_name = self.predict_model_var.get()

        try:
            # Preprocess
            cleaned = self.preprocessor.clean_text(text)

            # Extract features
            features = self.feature_extractor.transform([cleaned])

            # Predict
            prediction = self.trainer.predict(features, model_name)[0]

            try:
                proba = self.trainer.predict_proba(features, model_name)[0]
                fake_prob = proba[1]
                real_prob = proba[0]
            except Exception:
                fake_prob = None
                real_prob = None

            # Display result
            if prediction == 0:
                self.result_label.config(
                    text="✅ REAL NEWS",
                    foreground='green',
                    font=('Segoe UI', 18, 'bold')
                )
            else:
                self.result_label.config(
                    text="❌ FAKE NEWS",
                    foreground='red',
                    font=('Segoe UI', 18, 'bold')
                )

            # Detail text
            detail = f"Model Used: {model_name}\n"
            detail += f"{'=' * 50}\n\n"
            if fake_prob is not None:
                detail += f"Confidence Scores:\n"
                detail += f"  Real: {real_prob:.2%}\n"
                detail += f"  Fake: {fake_prob:.2%}\n\n"
            detail += f"Processed Text:\n{cleaned[:500]}\n"

            self.result_detail.delete('1.0', 'end')
            self.result_detail.insert('1.0', detail)

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def _clear_prediction(self):
        """Clear prediction inputs and results."""
        self.predict_text.delete('1.0', 'end')
        self.result_label.config(text="Enter text and click Analyze",
                                  foreground='black', font=('Segoe UI', 14))
        self.result_detail.delete('1.0', 'end')


def launch_gui():
    """Launch the GUI application."""
    root = tk.Tk()
    app = FakeNewsDetectorApp(root)
    root.mainloop()
