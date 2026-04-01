"""
=============================================================================
SPECTRALOGIC AI - MASTER BENCHMARKING & VALIDATION SUITE v10.3 (REDUCED)
Keeps only: Neuromorphic, Modified Cosine, Jaccard, DreaMS (3MS)
=============================================================================
Changes in this version:
  • Removed all other algorithms; only four selected remain.
  • Removed AI training data collection and p‑value test (dependencies removed).
  • All other features (GUI, metrics, exports) unchanged.
=============================================================================
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import math
import re
import os
import pandas as pd
import numpy as np
import cmath
import threading
import time
import json
import warnings
from collections import defaultdict, Counter
from datetime import datetime
import statistics
import sys

# =========================
# PATCH TKINTER VARIABLE.__del__ TO AVOID EXIT ERRORS
# =========================
try:
    import tkinter
    def _tkvariable_del(self):
        try:
            if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
                self._tk.globalunsetvar(self._name)
        except (RuntimeError, tkinter.TclError):
            pass
    tkinter.Variable.__del__ = _tkvariable_del
except:
    pass

# =========================
# OPTIONAL DEPENDENCIES – GRACEFUL DEGRADATION
# =========================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib/seaborn not installed. Graphics exports will be disabled.")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    warnings.warn("psutil not installed. Memory tracking will be disabled.")

try:
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    from sklearn.utils import resample
    from statsmodels.stats.contingency_tables import mcnemar
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not installed. ROC-AUC, F1, p‑value will be disabled.")

# =========================
# 1. STREAMING MSP PARSER (unchanged)
# =========================

def parse_msp_streaming(path):
    """
    Generator that yields one spectrum at a time from an MSP file.
    Every record always contains: name, inchikey, formula, precursormz, peaks, raw.
    """
    with open(path, encoding="utf-8", errors="ignore") as f:
        record = {
            "name": "N/A",
            "inchikey": "N/A",
            "formula": "N/A",
            "precursormz": 0.0,
            "peaks": [],
            "raw": ""
        }
        for line in f:
            stripped = line.strip()
            if not stripped:
                if record["peaks"]:
                    # Post-process
                    record["peaks"].sort(key=lambda x: x[1], reverse=True)
                    record["base_peak"] = record["peaks"][0]
                    record["max_mz"] = max(p[0] for p in record["peaks"])
                    record["tic"] = sum(p[1] for p in record["peaks"])
                    record["entropy"] = calculate_spectral_entropy(record["peaks"])
                    yield record
                # Reset for next record
                record = {
                    "name": "N/A",
                    "inchikey": "N/A",
                    "formula": "N/A",
                    "precursormz": 0.0,
                    "peaks": [],
                    "raw": ""
                }
                continue
            record["raw"] += line
            if ":" in stripped:
                header, data = [i.strip() for i in stripped.split(":", 1)]
                hl = header.lower()
                if hl == "name":
                    record["name"] = data
                elif hl == "inchikey":
                    record["inchikey"] = data
                elif hl == "formula":
                    record["formula"] = data
                elif hl == "precursormz":
                    try:
                        record["precursormz"] = float(data)
                    except:
                        pass
            elif re.match(r"^\d", stripped):
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        record["peaks"].append((float(parts[0]), float(parts[1])))
                    except:
                        pass
        if record["peaks"]:
            record["peaks"].sort(key=lambda x: x[1], reverse=True)
            record["base_peak"] = record["peaks"][0]
            record["max_mz"] = max(p[0] for p in record["peaks"])
            record["tic"] = sum(p[1] for p in record["peaks"])
            record["entropy"] = calculate_spectral_entropy(record["peaks"])
            yield record

def load_all_spectra(path):
    """Convenience: load all spectra into list (only for small/medium files)."""
    return list(parse_msp_streaming(path))

# =========================
# 2. COMPUTATIONAL UTILITIES (unchanged)
# =========================

def calculate_spectral_entropy(peaks):
    if not peaks: return 0.0
    total_i = sum(p[1] for p in peaks)
    if total_i == 0: return 0.0
    entropy = 0.0
    for _, inty in peaks:
        p = inty / total_i
        if p > 0: entropy -= p * math.log(p)
    return entropy

def normalize_l2(peaks):
    denom = math.sqrt(sum(p[1]**2 for p in peaks))
    if denom == 0: return peaks
    return [(p[0], p[1]/denom) for p in peaks]

def advanced_peak_matcher(q_peaks, l_peaks, tol):
    qn = normalize_l2(q_peaks)
    ln = normalize_l2(l_peaks)
    total_q = sum(p[1] for p in q_peaks)
    total_l = sum(p[1] for p in l_peaks)
    matched_q_raw, matched_l_raw = 0.0, 0.0
    matches = []
    used_l = set()
    mz_diffs = []
    for i, (mq, iq) in enumerate(qn):
        best_l, best_dist = -1, tol + 0.001
        for j, (ml, il) in enumerate(ln):
            if j in used_l: continue
            dist = abs(mq - ml)
            if dist <= tol and dist < best_dist:
                best_dist, best_l = dist, j
        if best_l != -1:
            matches.append({
                'mz': mq, 'q_i': iq, 'l_i': ln[best_l][1],
                'rank_q': i, 'rank_l': best_l
            })
            matched_q_raw += q_peaks[i][1]
            matched_l_raw += l_peaks[best_l][1]
            used_l.add(best_l)
            mz_diffs.append(abs(mq - ln[best_l][0]))
    f_tic = matched_q_raw / total_q if total_q > 0 else 0
    r_tic = matched_l_raw / total_l if total_l > 0 else 0
    if matches:
        avg_rank_diff = statistics.mean(abs(m['rank_q'] - m['rank_l']) for m in matches)
        high_int_matches = sum(1 for m in matches if m['q_i'] > 0.5 and m['l_i'] > 0.5)
        std_mz_diff = np.std(mz_diffs) if mz_diffs else 0
        low_mz_match_count = sum(1 for m in matches if m['mz'] < 100)
    else:
        avg_rank_diff = 0
        high_int_matches = 0
        std_mz_diff = 0
        low_mz_match_count = 0
    return matches, f_tic, r_tic, avg_rank_diff, high_int_matches, std_mz_diff, low_mz_match_count

# =========================
# 3. SELECTED ALGORITHMS (only the four requested)
# =========================

def neuromorphic_algorithm_v7(q_peaks, l_peaks, tol):
    """
    Neuromorphic algorithm.
    Theoretical intent: Spike‑timing‑dependent plasticity (STDP) inspired: rewards
    matches with similar ranks (temporal coincidence) and high intensity (strong synapses).
    """
    matches, q_tic, l_tic, avg_rank_diff, _, _, _ = advanced_peak_matcher(q_peaks, l_peaks, tol)
    M = len(matches)
    if M < 2:
        return 0
    spike_sum = 0.0
    for m in matches:
        spike_sum += (m['q_i'] * m['l_i']) * math.exp(-abs(m['rank_q'] - m['rank_l']) / 1.5)
    hebbian = sum(1 for m in matches if m['q_i'] > 0.6 and m['l_i'] > 0.6) * 0.3
    potential = spike_sum * (M / (M + 1)) * (1 + hebbian)
    score = potential * q_tic * l_tic
    return score * 10.0

def modified_cosine(q_peaks, l_peaks, tol):
    matches, _, _, _, _, _, _ = advanced_peak_matcher(q_peaks, l_peaks, tol)
    if not matches: return 0
    weighted = 0.0
    for m in matches:
        weighted += m['q_i'] * m['l_i']
    return weighted

def jaccard_spectral(q_peaks, l_peaks, tol):
    bins_q = set(int(mz / tol) for mz, _ in q_peaks)
    bins_l = set(int(mz / tol) for mz, _ in l_peaks)
    intersection = len(bins_q & bins_l)
    union = len(bins_q | bins_l)
    return intersection / union if union > 0 else 0

def dreams_similarity(q_peaks, l_peaks, tol):
    matches, q_tic, l_tic, _, _, _, _ = advanced_peak_matcher(q_peaks, l_peaks, tol)
    M = len(matches)
    if M == 0: return 0
    return (M / (len(q_peaks) + len(l_peaks))) * q_tic * l_tic * 2.0

# =============================================================================
# 4. ADVANCED METRICS – CATEGORY 2 (Accuracy & Reliability)
# =============================================================================

def compute_roc_auc_ci(scores, labels, n_bootstrap=1000, ci=95):
    """Return AUC and (lower, upper) confidence intervals."""
    if not HAS_SKLEARN or len(set(labels)) < 2:
        return 0.0, (0.0, 0.0)
    auc = roc_auc_score(labels, scores)
    bootstrapped_aucs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        indices = rng.choice(len(scores), len(scores), replace=True)
        if len(set(labels[indices])) < 2:
            continue
        boot_auc = roc_auc_score(labels[indices], scores[indices])
        bootstrapped_aucs.append(boot_auc)
    if len(bootstrapped_aucs) < 2:
        return auc, (auc, auc)
    lower = np.percentile(bootstrapped_aucs, (100-ci)/2)
    upper = np.percentile(bootstrapped_aucs, 100 - (100-ci)/2)
    return auc, (lower, upper)

# =============================================================================
# 5. COMPUTATIONAL EFFICIENCY – CATEGORY 3 (measured)
# =============================================================================

def measure_total_time(func, q_data, l_data, mz_tol, precursor_tol=0.0):
    """Measure total time (seconds) to compare all queries against all libraries."""
    start = time.perf_counter()
    for q in q_data:
        for lib in l_data:
            if precursor_tol > 0 and q["precursormz"] != 0 and lib["precursormz"] != 0:
                if abs(q["precursormz"] - lib["precursormz"]) > precursor_tol:
                    continue
            _ = func(q["peaks"], lib["peaks"], mz_tol)
    end = time.perf_counter()
    return end - start

# =============================================================================
# 6. SUSTAINABILITY & ECOLOGICAL METRICS – CATEGORY 4 (based on measured time)
# =============================================================================

def estimate_energy_from_time(total_time_sec, power_watt=50):
    """Estimate energy in Joules from total computation time and assumed CPU power."""
    return total_time_sec * power_watt

def carbon_footprint(joules, emission_factor=0.000233):
    kwh = joules / 3.6e6
    co2_kg = kwh * emission_factor * 1000
    return co2_kg

# =============================================================================
# 7. PUBLICATION GRAPHICS – CATEGORY 5 (FAIR compliant)
# =============================================================================

def sanitize_filename(s):
    return re.sub(r'[^\w\-_]', '_', s)

def export_mirror_plot(q_peaks, l_peaks, title, filename):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    q_mz = [p[0] for p in q_peaks]
    q_int = [p[1] for p in q_peaks]
    ax.bar(q_mz, q_int, width=0.5, label='Query', color='blue', alpha=0.7)
    l_mz = [p[0] for p in l_peaks]
    l_int = [-p[1] for p in l_peaks]
    ax.bar(l_mz, l_int, width=0.5, label='Library', color='red', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('m/z')
    ax.set_ylabel('Intensity (query positive, library negative)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def export_score_histogram(scores_correct, scores_incorrect, title, filename):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(scores_correct, bins=50, alpha=0.5, label='Correct', color='green')
    ax.hist(scores_incorrect, bins=50, alpha=0.5, label='Incorrect', color='red')
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# =============================================================================
# 8. FAIR COMPLIANCE – METADATA EXPORT (unchanged)
# =============================================================================

def export_fair_metadata(results, filename="fair_metadata.json"):
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "software": "Spectralogic AI v10.3 (Reduced)",
        "python_version": sys.version,
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__ if HAS_MPL else "not installed",
            "seaborn": sns.__version__ if HAS_MPL and 'seaborn' in sys.modules else "not installed",
            "psutil": psutil.__version__ if HAS_PSUTIL else "not installed",
            "sklearn": "installed" if HAS_SKLEARN else "not installed"
        },
        "parameters": {
            "mz_tolerance": results.get("mz_tol"),
            "precursor_tolerance": results.get("precursor_tol"),
            "cutoff": results.get("cutoff")
        },
        "results": results.get("summary", [])
    }
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2)

# =============================================================================
# 9. CORE BENCHMARK PIPELINE (reduced – no AI training, no p‑value)
# =============================================================================

class BenchmarkRunner:
    def __init__(self, q_data, l_data, mz_tol, precursor_tol, cutoff, selected_algorithms,
                 save_raw,
                 enable_cat2, enable_cat3, enable_cat4, enable_cat5,
                 progress_callback, cancel_flag):
        self.q_data = q_data
        self.l_data = l_data
        self.mz_tol = mz_tol
        self.precursor_tol = precursor_tol
        self.cutoff = cutoff
        self.selected = selected_algorithms
        self.save_raw = save_raw
        self.enable_cat2 = enable_cat2
        self.enable_cat3 = enable_cat3
        self.enable_cat4 = enable_cat4
        self.enable_cat5 = enable_cat5
        self.progress = progress_callback
        self.cancel = cancel_flag

        self.raw_results = []
        self.summary = []
        self.latency_data = {}
        self.peak_memory_data = {}

        self.process = psutil.Process() if HAS_PSUTIL else None

    def run(self):
        # Identify queries that have ground truth in library
        lib_inchikeys = {lib["inchikey"] for lib in self.l_data if lib["inchikey"] != "N/A"}
        total_queries = 0
        query_indices = []  # list of query indices that have ground truth
        for idx, q in enumerate(self.q_data):
            if q["inchikey"] != "N/A" and q["inchikey"] in lib_inchikeys:
                total_queries += 1
                query_indices.append(idx)

        for alg_idx, (label, algo) in enumerate(self.selected.items()):
            if self.cancel.is_set():
                break
            self.progress(alg_idx, len(self.selected), f"Running {label}...")

            TP = 0
            FP = 0
            FN = 0
            top_scores = []
            top_labels = []
            rank_correct_list = []   # store rank of correct match per query (None if not found)

            # Track memory
            if HAS_PSUTIL:
                self.process.cpu_percent()  # reset
                mem_start = self.process.memory_info().rss / 1024 / 1024
                peak_mem = mem_start

            # Time measurement (total for all comparisons)
            start_time = time.perf_counter()

            # Main matching loop
            for q_idx in query_indices:
                if self.cancel.is_set():
                    break

                query = self.q_data[q_idx]

                all_scores = []   # list of (score, lib_inchikey)

                for library in self.l_data:
                    # Optional precursor filter
                    if self.precursor_tol > 0:
                        if query["precursormz"] != 0.0 and library["precursormz"] != 0.0:
                            if abs(query["precursormz"] - library["precursormz"]) > self.precursor_tol:
                                continue
                    s = algo(query["peaks"], library["peaks"], self.mz_tol)
                    all_scores.append((s, library["inchikey"]))

                if not all_scores:
                    # No library passed the filter – treat as no hit
                    top_scores.append(0.0)
                    top_labels.append(0)
                    FN += 1
                    rank_correct_list.append(None)
                    continue

                # Sort descending by score
                all_scores.sort(reverse=True, key=lambda x: x[0])
                best_score, best_inchikey = all_scores[0]

                # For ROC, use top score and label
                is_correct = (best_inchikey == query["inchikey"])
                top_scores.append(best_score)
                top_labels.append(1 if is_correct else 0)

                # Determine TP/FP based on cutoff
                if best_score >= self.cutoff:
                    if is_correct:
                        TP += 1
                    else:
                        FP += 1
                else:
                    # Score below cutoff: if correct, it's FN; if incorrect, it's not counted (but for FN we count only correct)
                    if is_correct:
                        FN += 1

                # Find rank of correct match
                rank_correct = None
                for rank, (sc, ink) in enumerate(all_scores, start=1):
                    if ink == query["inchikey"]:
                        rank_correct = rank
                        break
                rank_correct_list.append(rank_correct)

                # Raw outcome for logging
                self.raw_results.append({
                    "Method": label,
                    "Query": query["name"],
                    "Match": best_inchikey,
                    "Sim": round(best_score, 3),
                    "Validation": "Success" if is_correct and best_score>=self.cutoff else "Fail"
                })

                # Memory tracking during loop
                if HAS_PSUTIL and q_idx % 100 == 0:
                    mem_current = self.process.memory_info().rss / 1024 / 1024
                    if mem_current > peak_mem:
                        peak_mem = mem_current

            # End of query loop
            end_time = time.perf_counter()
            total_time = end_time - start_time

            if HAS_PSUTIL:
                self.peak_memory_data[label] = peak_mem

            # Compute metrics
            total_q = total_queries
            recall = TP / total_q if total_q > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Top-k recall
            top1_recall = sum(1 for r in rank_correct_list if r == 1) / total_q if total_q > 0 else 0
            top3_recall = sum(1 for r in rank_correct_list if r is not None and r <= 3) / total_q if total_q > 0 else 0
            top10_recall = sum(1 for r in rank_correct_list if r is not None and r <= 10) / total_q if total_q > 0 else 0

            # ROC-AUC and CI
            auc, (auc_low, auc_high) = compute_roc_auc_ci(np.array(top_scores), np.array(top_labels))

            # Build summary row
            row = {
                "Method": label,
                "Total_Queries": total_q,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "Recall": round(recall, 3),
                "Precision": round(precision, 3),
                "F1_Score": round(f1, 3),
                "Top1_Recall": round(top1_recall, 3),
                "Top3_Recall": round(top3_recall, 3),
                "Top10_Recall": round(top10_recall, 3),
                "ROC_AUC": round(auc, 3),
                "ROC_AUC_95CI_low": round(auc_low, 3),
                "ROC_AUC_95CI_high": round(auc_high, 3)
            }

            if self.enable_cat3:
                row["Latency_ms"] = round(total_time * 1000, 3)
                row["Throughput_spectra_per_s"] = round(total_q / total_time if total_time>0 else 0, 1)
                if HAS_PSUTIL:
                    row["Peak_RAM_MB"] = round(self.peak_memory_data.get(label, 0), 1)

            if self.enable_cat4:
                energy_j = estimate_energy_from_time(total_time)
                co2 = carbon_footprint(energy_j)
                row["Energy_J_est"] = round(energy_j, 6)
                row["CO2_kg_est"] = round(co2, 6)

            # Scientific Confidence
            row["Scientific_Confidence"] = round(recall * precision, 3)

            self.summary.append(row)
            self._save_summary_incremental()

            # Category 5 graphics
            if self.enable_cat5 and HAS_MPL and rank_correct_list:
                safe_label = sanitize_filename(label)
                # Find a correct match for mirror plot
                for q_idx, rank in zip(query_indices, rank_correct_list):
                    if rank == 1:
                        q_spec = self.q_data[q_idx]
                        # Find a library with matching inchikey
                        for lib in self.l_data:
                            if lib["inchikey"] == q_spec["inchikey"]:
                                export_mirror_plot(q_spec["peaks"], lib["peaks"],
                                                   f"{label} Mirror Plot", f"mirror_{safe_label}.png")
                                break
                        break
                # Score histogram
                correct_scores = [s for s, l in zip(top_scores, top_labels) if l == 1]
                incorrect_scores = [s for s, l in zip(top_scores, top_labels) if l == 0]
                export_score_histogram(correct_scores, incorrect_scores,
                                       f"{label} Score Distribution", f"hist_{safe_label}.png")

        # Final exports
        if self.save_raw and not self.cancel.is_set():
            pd.DataFrame(self.raw_results).to_csv("benchmark_raw_outcomes.csv", index=False)

        if self.enable_cat5:
            export_fair_metadata({
                "mz_tol": self.mz_tol,
                "precursor_tol": self.precursor_tol,
                "cutoff": self.cutoff,
                "summary": self.summary
            })

        self.progress(len(self.selected), len(self.selected), "Done.")

    def _save_summary_incremental(self):
        df = pd.DataFrame(self.summary)
        df.to_csv("benchmark_summary.csv", index=False)

# =============================================================================
# 10. GRAPHICAL USER INTERFACE (simplified algorithm selection)
# =============================================================================

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.bind_mousewheel()

    def bind_mousewheel(self):
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("SENTRY Spectral Engine v10.3 - Reduced (4 Algorithms)")
        self.root.geometry("1400x1050")
        self.path_q = tk.StringVar()
        self.path_l = tk.StringVar()
        self.var_save_raw = tk.BooleanVar(value=False)
        self.var_cat2 = tk.BooleanVar(value=True)
        self.var_cat3 = tk.BooleanVar(value=True)
        self.var_cat4 = tk.BooleanVar(value=True)
        self.var_cat5 = tk.BooleanVar(value=True)
        self.cancel_flag = threading.Event()
        self.runner = None
        self.benchmark_thread = None

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill="both", expand=True)
        self.scrollable = ScrollableFrame(self.main_container)
        self.scrollable.pack(fill="both", expand=True)
        main = self.scrollable.scrollable_frame
        main.columnconfigure(0, weight=1)

        header = ttk.Label(main, text="SPECTRALOGIC AI - REDUCED BENCHMARK SUITE (4 Algorithms)",
                           font=("Segoe UI", 14, "bold"))
        header.grid(row=0, column=0, pady=(10,20), sticky="ew")

        # File selection
        file_frame = ttk.LabelFrame(main, text="Input Files", padding=10)
        file_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="Query Dataset (.msp):").grid(row=0, column=0, sticky="w")
        ttk.Entry(file_frame, textvariable=self.path_q, width=60).grid(row=0, column=1, padx=10, sticky="ew")
        ttk.Button(file_frame, text="SELECT", command=self.set_query).grid(row=0, column=2)

        ttk.Label(file_frame, text="Library Reference(s):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(file_frame, textvariable=self.path_l, width=60).grid(row=1, column=1, padx=10, sticky="ew")
        ttk.Button(file_frame, text="SELECT", command=self.set_lib).grid(row=1, column=2)

        # Parameters
        param_frame = ttk.LabelFrame(main, text="Similarity Parameters", padding=10)
        param_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        param_inner = ttk.Frame(param_frame)
        param_inner.pack()

        ttk.Label(param_inner, text="m/z Mass Tolerance:").pack(side="left")
        self.e_tol = ttk.Entry(param_inner, width=10)
        self.e_tol.insert(0, "0.01")
        self.e_tol.pack(side="left", padx=10)

        ttk.Label(param_inner, text="Precursor Tolerance (Da, 0 to disable):").pack(side="left")
        self.e_pre_tol = ttk.Entry(param_inner, width=10)
        self.e_pre_tol.insert(0, "0.01")
        self.e_pre_tol.pack(side="left", padx=10)

        ttk.Label(param_inner, text="Min Score Cutoff:").pack(side="left")
        self.e_cutoff = ttk.Entry(param_inner, width=10)
        self.e_cutoff.insert(0, "0.5")
        self.e_cutoff.pack(side="left", padx=10)

        # Advanced Metrics Category Selection
        metrics_frame = ttk.LabelFrame(main, text="Advanced Metrics (Categories 2‑5)", padding=10)
        metrics_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        metrics_inner = ttk.Frame(metrics_frame)
        metrics_inner.pack(fill="x", expand=True)

        ttk.Checkbutton(metrics_inner, text="Category 2: Accuracy & Reliability (Top‑N, ROC with CI, F1)",
                        variable=self.var_cat2).grid(row=0, column=0, sticky="w", padx=20)
        ttk.Checkbutton(metrics_inner, text="Category 3: Computational Efficiency (Latency, Throughput, Memory)",
                        variable=self.var_cat3).grid(row=1, column=0, sticky="w", padx=20)
        ttk.Checkbutton(metrics_inner, text="Category 4: Sustainability (Energy, CO2)",
                        variable=self.var_cat4).grid(row=2, column=0, sticky="w", padx=20)
        ttk.Checkbutton(metrics_inner, text="Category 5: Publication Graphics & FAIR Compliance",
                        variable=self.var_cat5).grid(row=3, column=0, sticky="w", padx=20)

        # Algorithm Selection (only 4)
        algo_container = ttk.LabelFrame(main, text="Algorithm Selection (4 total)", padding=10)
        algo_container.grid(row=4, column=0, sticky="ew", padx=10, pady=10)
        algo_frame = ttk.Frame(algo_container)
        algo_frame.pack()

        self.algo_vars = {}
        algos = ["Neuromorphic", "Modified Cosine", "Jaccard", "DreaMS (3MS)"]
        for i, name in enumerate(algos):
            var = tk.BooleanVar(value=True)
            self.algo_vars[name] = var
            cb = ttk.Checkbutton(algo_frame, text=name, variable=var)
            cb.grid(row=0, column=i, padx=20, pady=10)

        # Export Options
        export_frame = ttk.LabelFrame(main, text="Export Options", padding=10)
        export_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=10)

        ttk.Checkbutton(export_frame, text="Export benchmark_raw_outcomes.csv", variable=self.var_save_raw).grid(
            row=0, column=0, sticky="w", padx=20)

        # Progress and status
        progress_frame = ttk.LabelFrame(main, text="Progress", padding=10)
        progress_frame.grid(row=6, column=0, sticky="ew", padx=10, pady=10)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=600)
        self.progress_bar.pack(pady=5)
        self.status_label = ttk.Label(progress_frame, text="Ready.", font=("Segoe UI", 9))
        self.status_label.pack()

        # Buttons
        button_frame = ttk.Frame(main)
        button_frame.grid(row=7, column=0, pady=20)

        self.start_btn = ttk.Button(button_frame, text="🚀 START BENCHMARK",
                                    command=self.start_benchmark, width=30)
        self.start_btn.pack(side="left", padx=10)

        self.cancel_btn = ttk.Button(button_frame, text="⛔ CANCEL",
                                     command=self.cancel_benchmark, state="disabled", width=20)
        self.cancel_btn.pack(side="left", padx=10)

        footer = ttk.Label(main, text="v10.3 – Reduced to 4 algorithms (Neuromorphic, Modified Cosine, Jaccard, DreaMS)",
                           font=("Segoe UI", 9, "italic"))
        footer.grid(row=8, column=0, pady=10)

    def set_query(self):
        f = filedialog.askopenfilename(filetypes=[("MSP Spectrum", "*.msp")])
        if f: self.path_q.set(f)

    def set_lib(self):
        f = filedialog.askopenfilenames(filetypes=[("MSP Library", "*.msp")])
        if f: self.path_l.set("; ".join(f))

    def update_progress(self, current, total, message):
        self.root.after(0, self._update_gui, current, total, message)

    def _update_gui(self, current, total, message):
        self.progress_var.set((current + 1) / total * 100)
        self.status_label.config(text=message)

    def start_benchmark(self):
        if not self.path_q.get() or not self.path_l.get():
            messagebox.showwarning("Incomplete", "Please select both Query and Library files.")
            return

        # Build selected algorithms dict (only the four)
        selected = {}
        for name, var in self.algo_vars.items():
            if var.get():
                if name == "Neuromorphic":
                    selected[name] = neuromorphic_algorithm_v7
                elif name == "Modified Cosine":
                    selected[name] = modified_cosine
                elif name == "Jaccard":
                    selected[name] = jaccard_spectral
                elif name == "DreaMS (3MS)":
                    selected[name] = dreams_similarity

        if not selected:
            messagebox.showwarning("No algorithms", "Please select at least one algorithm.")
            return

        # Parse data
        try:
            self.status_label.config(text="Parsing query spectra...")
            self.root.update()
            q_data = load_all_spectra(self.path_q.get())
            lib_paths = self.path_l.get().split("; ")
            l_data = []
            for p in lib_paths:
                l_data.extend(load_all_spectra(p.strip()))
        except Exception as e:
            messagebox.showerror("Parse Error", f"Failed to parse MSP files:\n{e}")
            return

        self.cancel_flag.clear()
        self.start_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")

        def task():
            runner = BenchmarkRunner(
                q_data, l_data,
                float(self.e_tol.get()),
                float(self.e_pre_tol.get()),
                float(self.e_cutoff.get()),
                selected,
                self.var_save_raw.get(),
                self.var_cat2.get(),
                self.var_cat3.get(),
                self.var_cat4.get(),
                self.var_cat5.get(),
                self.update_progress,
                self.cancel_flag
            )
            self.runner = runner
            runner.run()
            self.root.after(0, self.benchmark_finished)

        self.benchmark_thread = threading.Thread(target=task, daemon=True)
        self.benchmark_thread.start()

    def cancel_benchmark(self):
        self.cancel_flag.set()
        self.status_label.config(text="Cancelling...")
        self.cancel_btn.config(state="disabled")

    def benchmark_finished(self):
        self.start_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        if self.cancel_flag.is_set():
            self.status_label.config(text="Cancelled. Partial results saved.")
        else:
            self.status_label.config(text="Benchmark complete. All results saved.")
        self.benchmark_thread = None
        messagebox.showinfo("Finished", "Benchmark completed.\nSee benchmark_summary.csv, plots, and fair_metadata.json")

    def on_closing(self):
        if self.benchmark_thread is not None and self.benchmark_thread.is_alive():
            self.cancel_flag.set()
            self.benchmark_thread.join(timeout=2)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    try:
        root.mainloop()
    finally:
        # Clean up references to help avoid Tkinter variable __del__ errors
        app = None
        root = None