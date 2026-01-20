import sys
import os
import glob
import json
import logging
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.gridspec import GridSpec

from scipy.special import erf
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QMessageBox,
    QLabel, QDoubleSpinBox, QProgressBar, QSpinBox,
    QComboBox, QCheckBox, QGroupBox, QGridLayout, QDockWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".tof_explorer_config.json")

DEFAULT_SETTINGS = {
    "calibration": {
        "TOF_OFFSET_NS": 0.0,
        "WORK_FUNCTION_EV": 0.0,
        "FLIGHT_DISTANCE_M": 0.768,
    },
    "plots": {
        "Raw Avg": {"cmap": "viridis", "vmin": 0.0, "vmax": 0.4,
                    "slice": {"xmin": None, "xmax": None, "colmin": None, "colmax": None},
                    "ylim": {"ymin": None, "ymax": None}},
        "Folded": {"cmap": "viridis", "vmin": 0.0, "vmax": 0.4,
                   "slice": {"xmin": None, "xmax": None, "colmin": None, "colmax": None},
                   "ylim": {"ymin": None, "ymax": None}},
        "SC Corrected": {"cmap": "viridis", "vmin": 0.0, "vmax": 0.4,
                         "slice": {"xmin": None, "xmax": None, "colmin": None, "colmax": None},
                         "ylim": {"ymin": None, "ymax": None}},
        "FFT": {"slice": {"xmin": None, "xmax": None, "colmin": None, "colmax": None}},
        "Dynamics Log": {"ylim": {"ymin": None, "ymax": None}},
        "Dynamics Lin": {"ylim": {"ymin": None, "ymax": None}},
        "Residuals": {"ylim": {"ymin": None, "ymax": None}},
    },
    "fit": {
        "t0_fixed_mm": 142.298,
        "t0_optimisation": False,
    },
    "data": {
        "POINTS_PER_NS": 1.0 / 0.8,
        "BIN_TO_NS_FLAG": False,
    },
    "ui": {
        "colormaps": ["viridis", "plasma", "inferno", "magma", "jet", "gray", "cividis"],
        "last_folder": "",
        "auto_watch": False,
    },
}


def load_settings():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
            merged = DEFAULT_SETTINGS.copy()
            for k, v in DEFAULT_SETTINGS.items():
                if isinstance(v, dict):
                    merged[k] = {**v, **data.get(k, {})}
                else:
                    merged[k] = data.get(k, v)
            # deep-ish merge for plots
            if "plots" in data:
                for pname, psettings in data["plots"].items():
                    merged["plots"][pname] = {**merged["plots"].get(pname, {}), **psettings}
            return merged
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")
    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(settings, f, indent=2)
        logger.info(f"Settings saved to {CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")


def _safe_float(value, default=0.0):
    try:
        return float(value) if value is not None else float(default)
    except Exception:
        return float(default)


GLOBAL_SETTINGS = load_settings()


class PhysicalConstants:
    ELECTRON_MASS = 9.1093837e-31
    ELEMENTARY_CHARGE = 1.602176634e-19


class CalibrationConstants:
    def __init__(self, TOF_OFFSET_NS=None, WORK_FUNCTION_EV=None, FLIGHT_DISTANCE_M=None):
        self.TOF_OFFSET_NS = _safe_float(TOF_OFFSET_NS, GLOBAL_SETTINGS["calibration"]["TOF_OFFSET_NS"])
        self.WORK_FUNCTION_EV = _safe_float(WORK_FUNCTION_EV, GLOBAL_SETTINGS["calibration"]["WORK_FUNCTION_EV"])
        self.FLIGHT_DISTANCE_M = _safe_float(FLIGHT_DISTANCE_M, GLOBAL_SETTINGS["calibration"]["FLIGHT_DISTANCE_M"])


def tof_to_ke(tof_ns, calibration: CalibrationConstants, bin_to_ns=False, points_per_ns=None):
    arr = np.asarray(tof_ns, dtype=float)
    if bin_to_ns:
        pts = _safe_float(points_per_ns, GLOBAL_SETTINGS["data"].get("POINTS_PER_NS", 1.0/0.8))
        if pts == 0:
            pts = 1.0
        arr = arr / pts

    tof_shifted = arr - calibration.TOF_OFFSET_NS
    tof_shifted = np.where(np.abs(tof_shifted) < 1e-12, 1e-12, tof_shifted)

    v = calibration.FLIGHT_DISTANCE_M / (tof_shifted * 1e-9)
    ke_eV = 0.5 * PhysicalConstants.ELECTRON_MASS * v**2 / PhysicalConstants.ELEMENTARY_CHARGE
    # Note: old code subtracts a constant 0.51 in its expression; your newer code uses WORK_FUNCTION separately.
    return ke_eV


def tof_to_binding_energy(tof_ns, photon_energy_eV, calibration: CalibrationConstants, bin_to_ns=False, points_per_ns=None):
    ke = tof_to_ke(tof_ns, calibration, bin_to_ns=bin_to_ns, points_per_ns=points_per_ns)
    return _safe_float(photon_energy_eV) - ke - calibration.WORK_FUNCTION_EV


MAX_DISPLAY_COLS = 1400

class FastLoader(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    @staticmethod
    def _extract_file_number(filepath: str) -> int:
        """
        Numeric key like old OnlineAnalysis:
        int(a.split('_')[-1].split('.')[0])
        Robust for 'TOF_123.dat' and also 'TOF123.dat' (fallback regex).
        """
        base = os.path.basename(filepath)
        m = re.search(r"(\d+)", base)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return 0
        return 0

    def _read_tof_like_old(self, fpath):
        try:
            df = pd.read_table(fpath, comment="#", sep=" ", header=None, engine="python")
            return df
        except Exception:
            # robust fallback
            df = pd.read_table(fpath, comment="#", delim_whitespace=True, header=None, engine="python")
            return df

    def run(self):
        try:
            cache_file = os.path.join(self.folder_path, "processed_cache.npz")
            if os.path.exists(cache_file):
                self.progress.emit(10)
                try:
                    cache = np.load(cache_file)
                    self.progress.emit(40)
                    self.finished.emit({
                        "tof": cache["tof"],
                        "analog": cache["analog"],
                        "counting": cache["counting"],
                        "folder": self.folder_path,
                        "cached": True,
                    })
                    self.progress.emit(100)
                    return
                except Exception:
                    logger.warning("Cache exists but failed to load; falling back to parsing")

            files = glob.glob(os.path.join(self.folder_path, "TOF*.dat"))
            files = sorted(files, key=self._extract_file_number)
            if not files:
                raise FileNotFoundError("No TOF*.dat files found")

            analog_list, counting_list = [], []
            tof_axis = None
            total = len(files)

            for i, fpath in enumerate(files):
                try:
                    df = self._read_tof_like_old(fpath)
                    if df.shape[1] < 2:
                        logger.warning(f"{fpath} has fewer than 2 columns — skipping")
                        continue
                    if tof_axis is None:
                        tof_axis = df.iloc[:, 0].to_numpy()
                    analog_list.append(df.iloc[:, 1].to_numpy())
                    if df.shape[1] > 2:
                        counting_list.append(df.iloc[:, 2].to_numpy())
                    else:
                        counting_list.append(np.zeros_like(tof_axis))
                except Exception as e_file:
                    logger.warning(f"Failed to parse {fpath}: {e_file} — skipping")
                    continue

                self.progress.emit(int(80 * (i + 1) / total))

            if not analog_list:
                raise RuntimeError("No valid TOF data files could be parsed")

            analog_arr = np.vstack(analog_list)
            counting_arr = np.vstack(counting_list)

            np.savez_compressed(cache_file, tof=tof_axis, analog=analog_arr, counting=counting_arr)

            self.progress.emit(90)
            self.finished.emit({
                "tof": tof_axis,
                "analog": analog_arr,
                "counting": counting_arr,
                "folder": self.folder_path,
                "cached": False,
            })
            self.progress.emit(100)
        except Exception as e:
            logger.exception("FastLoader failed")
            self.finished.emit({"error": str(e)})

class AnalysisWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)

    def __init__(self, folder, data, params):
        super().__init__()
        self.folder = folder
        self.data = data
        self.params = params

    @staticmethod
    def fold_twoway(M, pts):
        sx, sy = M.shape
        if pts <= 0:
            return np.zeros((0, sy))
        n = sx // pts
        fM = np.zeros((pts, sy))
        for i in range(n):
            scan = M[i*pts:(i+1)*pts, :]
            fM += scan if i % 2 == 0 else np.flipud(scan)
        return fM

    @staticmethod
    def pump_charge_edge(tau, a, b, c):
        b = np.maximum(b, 1e-10)
        return np.where(tau < 0, c, a*tau/(b+tau) + c)

    @staticmethod
    def find_pump_charge(tau, M, level=30):
        if M.size == 0:
            return np.zeros(M.shape[0], dtype=float), np.zeros_like(tau)
        n_rows, n_cols = M.shape
        edges = np.zeros(n_rows, dtype=float)
        for i in range(n_rows):
            row = M[i, :]
            idx = np.where(row >= level)[0]
            if idx.size == 0:
                k = int(np.argmax(row))
                edges[i] = float(k)
            else:
                k = idx[0]
                if k == 0:
                    edges[i] = 0.0
                else:
                    y0, y1 = row[k-1], row[k]
                    frac = 0.0 if (y1 == y0) else (level - y0) / (y1 - y0)
                    edges[i] = (k - 1) + float(frac)
        try:
            p, _ = curve_fit(
                AnalysisWorker.pump_charge_edge,
                tau[:edges.size],
                edges,
                [10, 10000, 100],
                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                maxfev=5000,
            )
            fitted = AnalysisWorker.pump_charge_edge(tau, *p)
            return edges, fitted
        except Exception:
            return edges, np.zeros_like(tau)

    @staticmethod
    def shift_rows_fill_zero(M, shifts):
        M = np.asarray(M)
        n_rows, n_cols = M.shape
        out = np.zeros_like(M)
        shifts = np.asarray(shifts).astype(int)
        for i in range(min(n_rows, shifts.size)):
            s = shifts[i]
            if s == 0:
                out[i] = M[i]
            elif s > 0:
                out[i, s:] = M[i, :n_cols - s]
            else:
                sneg = -s
                out[i, :n_cols - sneg] = M[i, sneg:]
        return out

    def run(self):
        try:
            AVG = self.data["analog"]
            CNT = self.data["counting"]
            n_start = int(self.params.get("n_start", 0))
            n_stop = int(self.params.get("n_stop", AVG.shape[0]))
            n_pts = int(self.params.get("n_pts", 1))
            roi_min = int(self.params.get("roi_min", 295))
            roi_max = int(self.params.get("roi_max", 310))
            level = float(self.params.get("edge_level", 30))
            model_name = self.params.get("model", "two_exp1")
            t_fs = self.params.get("t_fs", np.zeros(0))
            fft_requested = bool(self.params.get("fft", True))

            self.progress.emit(0)
            fAVG = self.fold_twoway(AVG[n_start:n_stop], n_pts)
            fCNT = self.fold_twoway(CNT[n_start:n_stop], n_pts)
            self.progress.emit(15)

            col0 = max(0, min(290, fCNT.shape[1]-1))
            col1 = max(col0+1, min(325, fCNT.shape[1]))
            edge_positions, fitted_edge = self.find_pump_charge(t_fs, fCNT[:, col0:col1], level)
            self.progress.emit(35)

            ref = int(round(edge_positions[0])) if edge_positions.size > 0 else 0
            shifts = np.round(ref - edge_positions).astype(int)
            rfCNT = self.shift_rows_fill_zero(fCNT, shifts)
            rfAVG = self.shift_rows_fill_zero(fAVG, shifts)
            self.progress.emit(55)

            roi_min = max(0, min(roi_min, rfCNT.shape[1]-1))
            roi_max = max(roi_min+1, min(roi_max, rfCNT.shape[1]))
            S = np.sum(rfCNT[:, roi_min:roi_max], axis=1)
            self.progress.emit(65)

            if model_name == "one_exp":
                p0 = [0, 30, 1000, 30, 10, 5]
            elif model_name == "two_exp":
                p0 = [0, 30, 1000, 10000, 100, 40, 0, 5]
            else:
                p0 = [0, 30, 700, 7000, 100, 40, 5]
            p_full = np.asarray(p0, dtype=float)
            pcov = None
            self.progress.emit(80)

            fft_result = None
            if fft_requested:
                profile = np.mean(rfCNT[:, roi_min:roi_max], axis=0)
                tof_arr = self.params.get("tof_arr", None)
                dt_ns = 1.0
                if tof_arr is not None:
                    diffs = np.diff(tof_arr)
                    dt_ns = float(np.median(diffs)) if diffs.size > 0 else 1.0
                sig = profile - np.mean(profile)
                N = sig.size
                spec = np.fft.rfft(sig)
                freq = np.fft.rfftfreq(N, d=dt_ns)
                power = np.abs(spec)
                fft_result = {"freq_ghz": freq, "power": power}
            self.progress.emit(90)

            out = {
                "fAVG": fAVG,
                "fCNT": fCNT,
                "rfCNT": rfCNT,
                "rfAVG": rfAVG,
                "edge_positions": edge_positions,
                "fitted_edge": fitted_edge,
                "S": S,
                "p": p_full,
                "pcov": pcov,
                "t_fs": t_fs,
                "fft": fft_result,
                "model_name": model_name,
            }
            self.progress.emit(100)
            self.finished.emit(out)
        except Exception as e:
            logger.exception("AnalysisWorker failed")
            self.finished.emit({"error": str(e)})

class AnalysisWindow(QMainWindow):
    IMAGE_PLOTS = ["Raw Avg", "Folded", "SC Corrected"]
    LINE_PLOTS = ["FFT", "Dynamics Log", "Dynamics Lin", "Residuals"]
    ALL_PLOTS = IMAGE_PLOTS + LINE_PLOTS

    @staticmethod
    def one_exp(t, t0, sig, t1, A1, A3, B):
        dt = t - t0
        sig = np.maximum(sig, 1e-10)
        t1 = np.maximum(t1, 1e-10)
        return A1*0.5*(1+erf((dt/sig - sig/t1)/np.sqrt(2)))*np.exp(-dt/t1) + A3*0.5*(1+erf(dt/sig/np.sqrt(2))) + B

    @staticmethod
    def two_exp1(t, t0, sig, t1, t2, A1, A2, B):
        dt = t - t0
        sig = np.maximum(sig, 1e-10)
        t1, t2 = np.maximum(t1, 1e-10), np.maximum(t2, 1e-10)
        return A1*0.5*(1+erf((dt/sig - sig/t1)/np.sqrt(2)))*np.exp(-dt/t1) + A2*0.5*(1+erf((dt/sig - sig/t2)/np.sqrt(2)))*np.exp(-dt/t2) + B

    @staticmethod
    def two_exp(t, t0, sig, t1, t2, A1, A2, A3, B):
        dt = t - t0
        sig = np.maximum(sig, 1e-10)
        t1, t2 = np.maximum(t1, 1e-10), np.maximum(t2, 1e-10)
        return A1*0.5*(1+erf((dt/sig - sig/t1)/np.sqrt(2)))*np.exp(-dt/t1) + A2*0.5*(1+erf((dt/sig - sig/t2)/np.sqrt(2)))*np.exp(-dt/t2) + A3*0.5*(1+erf(dt/sig/np.sqrt(2))) + B

    def __init__(self, folder, data, main_window=None):
        super().__init__()
        self.setWindowTitle(f"Analysis: {os.path.basename(folder)}")
        self.resize(1400, 900)
        self.folder = folder
        self.data = data
        self.main_window = main_window
        self.TOF = np.asarray(data["tof"])
        self._last_analysis = None
        self._analysis_worker = None

        self._plot_artists = {}
        self._axes_list = []
        self._artists_initialized = False
        self._orig_positions = None
        self._current_im_axes = {}
        self._pan_state = {}

        self._setup_ui()
        self._create_expert_dock()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        controls = self._create_controls()
        layout.addLayout(controls, 1)

        self.figure = plt.figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        layout.addWidget(self.canvas, 4)

    def _create_controls(self):
        v = QVBoxLayout()

        params = QGroupBox("Analysis parameters")
        p = QGridLayout()

        self.spin_l0 = QDoubleSpinBox()
        self.spin_l0.setRange(-1000, 1000)
        self.spin_l0.setDecimals(6)
        self.spin_l0.setValue(_safe_float(GLOBAL_SETTINGS["fit"].get("t0_fixed_mm", 142.298)))
        p.addWidget(QLabel("l0 (mm):"), 0, 0)
        p.addWidget(self.spin_l0, 0, 1)

        self.spin_nstart = QSpinBox()
        self.spin_nstart.setRange(0, 100000)
        self.spin_nstart.setValue(0)
        p.addWidget(QLabel("Start file index:"), 1, 0)
        p.addWidget(self.spin_nstart, 1, 1)

        self.spin_nstop = QSpinBox()
        self.spin_nstop.setRange(1, 100000)
        self.spin_nstop.setValue(min(2900, self.data["analog"].shape[0]))
        p.addWidget(QLabel("Stop file index:"), 2, 0)
        p.addWidget(self.spin_nstop, 2, 1)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["one_exp", "two_exp", "two_exp1"])
        self.model_combo.setCurrentText("two_exp1")
        p.addWidget(QLabel("Model:"), 3, 0)
        p.addWidget(self.model_combo, 3, 1)

        ncols = self.data["analog"].shape[1]
        self.spin_roi_min = QSpinBox()
        self.spin_roi_min.setRange(0, max(0, ncols-1))
        self.spin_roi_min.setValue(295 if ncols > 295 else 0)
        self.spin_roi_max = QSpinBox()
        self.spin_roi_max.setRange(1, max(1, ncols))
        self.spin_roi_max.setValue(310 if ncols > 310 else ncols)
        p.addWidget(QLabel("ROI col min:"), 4, 0)
        p.addWidget(self.spin_roi_min, 4, 1)
        p.addWidget(QLabel("ROI col max:"), 5, 0)
        p.addWidget(self.spin_roi_max, 5, 1)

        params.setLayout(p)
        v.addWidget(params)

        plots = QGroupBox("Show plots")
        pl = QVBoxLayout()
        self.chk_plots = {}
        for nm in self.ALL_PLOTS:
            cb = QCheckBox(nm)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_plot_visibility_changed)
            self.chk_plots[nm] = cb
            pl.addWidget(cb)
        plots.setLayout(pl)
        v.addWidget(plots)

        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        v.addWidget(self.btn_run)

        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        v.addStretch()
        return v

    def _create_expert_dock(self):
        dock = QDockWidget("Expert (Analysis)", self)
        w = QWidget()
        layout = QVBoxLayout(w)

        data_g = QGroupBox("Data / Sampling")
        dg = QGridLayout()
        self.spin_points_per_ns = QDoubleSpinBox()
        self.spin_points_per_ns.setDecimals(6)
        self.spin_points_per_ns.setRange(0.01, 1000)
        self.spin_points_per_ns.setValue(_safe_float(GLOBAL_SETTINGS["data"].get("POINTS_PER_NS", 1.0/0.8)))
        self.chk_bin_to_ns = QCheckBox("BIN_TO_NS_FLAG")
        self.chk_bin_to_ns.setChecked(bool(GLOBAL_SETTINGS["data"].get("BIN_TO_NS_FLAG", False)))
        dg.addWidget(QLabel("Points per ns:"), 0, 0)
        dg.addWidget(self.spin_points_per_ns, 0, 1)
        dg.addWidget(self.chk_bin_to_ns, 1, 0, 1, 2)
        data_g.setLayout(dg)
        layout.addWidget(data_g)

        cal_g = QGroupBox("Calibration & Fit")
        cl = QGridLayout()
        self.spin_tof_offset = QDoubleSpinBox()
        self.spin_tof_offset.setDecimals(6)
        self.spin_tof_offset.setRange(-1000, 1000)
        self.spin_tof_offset.setValue(_safe_float(GLOBAL_SETTINGS["calibration"]["TOF_OFFSET_NS"]))
        self.spin_workfunc = QDoubleSpinBox()
        self.spin_workfunc.setDecimals(4)
        self.spin_workfunc.setRange(-10, 10)
        self.spin_workfunc.setValue(_safe_float(GLOBAL_SETTINGS["calibration"]["WORK_FUNCTION_EV"]))
        self.spin_flightdist = QDoubleSpinBox()
        self.spin_flightdist.setDecimals(6)
        self.spin_flightdist.setRange(0.0, 10.0)
        self.spin_flightdist.setValue(_safe_float(GLOBAL_SETTINGS["calibration"]["FLIGHT_DISTANCE_M"]))
        self.spin_t0_mm = QDoubleSpinBox()
        self.spin_t0_mm.setDecimals(6)
        self.spin_t0_mm.setRange(-1000, 1000)
        self.spin_t0_mm.setValue(_safe_float(GLOBAL_SETTINGS["fit"].get("t0_fixed_mm", 142.298)))
        self.chk_t0_optim = QCheckBox("t0 optimisation")
        self.chk_t0_optim.setChecked(bool(GLOBAL_SETTINGS["fit"].get("t0_optimisation", False)))

        cl.addWidget(QLabel("TOF offset (ns):"), 0, 0)
        cl.addWidget(self.spin_tof_offset, 0, 1)
        cl.addWidget(QLabel("Work function (eV):"), 1, 0)
        cl.addWidget(self.spin_workfunc, 1, 1)
        cl.addWidget(QLabel("Flight dist (m):"), 2, 0)
        cl.addWidget(self.spin_flightdist, 2, 1)
        cl.addWidget(QLabel("t0 (mm):"), 3, 0)
        cl.addWidget(self.spin_t0_mm, 3, 1)
        cl.addWidget(self.chk_t0_optim, 4, 0, 1, 2)
        cal_g.setLayout(cl)
        layout.addWidget(cal_g)

        buttons = QHBoxLayout()
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self._apply_expert)
        self.btn_reset = QPushButton("Reset defaults")
        self.btn_reset.clicked.connect(self._reset_expert)
        self.btn_save_pdf = QPushButton("Save figure PDF")
        self.btn_save_pdf.clicked.connect(self._save_pdf)
        self.btn_export_npz = QPushButton("Export processed (.npz)")
        self.btn_export_npz.clicked.connect(self._export_processed)
        buttons.addWidget(self.btn_apply)
        buttons.addWidget(self.btn_reset)
        buttons.addWidget(self.btn_save_pdf)
        buttons.addWidget(self.btn_export_npz)

        layout.addLayout(buttons)
        layout.addStretch()
        dock.setWidget(w)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _apply_expert(self):
        GLOBAL_SETTINGS["data"]["POINTS_PER_NS"] = _safe_float(self.spin_points_per_ns.value())
        GLOBAL_SETTINGS["data"]["BIN_TO_NS_FLAG"] = bool(self.chk_bin_to_ns.isChecked())
        GLOBAL_SETTINGS["calibration"]["TOF_OFFSET_NS"] = _safe_float(self.spin_tof_offset.value())
        GLOBAL_SETTINGS["calibration"]["WORK_FUNCTION_EV"] = _safe_float(self.spin_workfunc.value())
        GLOBAL_SETTINGS["calibration"]["FLIGHT_DISTANCE_M"] = _safe_float(self.spin_flightdist.value())
        GLOBAL_SETTINGS["fit"]["t0_fixed_mm"] = _safe_float(self.spin_t0_mm.value())
        GLOBAL_SETTINGS["fit"]["t0_optimisation"] = bool(self.chk_t0_optim.isChecked())
        save_settings(GLOBAL_SETTINGS)
        QMessageBox.information(self, "Expert", "Settings applied and saved.")
        if self.main_window is not None:
            self.main_window.spin_view_tof_offset.setValue(_safe_float(GLOBAL_SETTINGS["calibration"]["TOF_OFFSET_NS"]))
            self.main_window.spin_view_workfunc.setValue(_safe_float(GLOBAL_SETTINGS["calibration"]["WORK_FUNCTION_EV"]))
            self.main_window._axis_mode_changed(force=True)
            self.main_window.update_plot()

    def _reset_expert(self):
        for k in DEFAULT_SETTINGS:
            GLOBAL_SETTINGS[k] = DEFAULT_SETTINGS[k]
        save_settings(GLOBAL_SETTINGS)
        QMessageBox.information(self, "Expert", "Settings reset to defaults. Restart app to fully reload defaults.")

    def _save_pdf(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save figure as PDF", "", "PDF files (*.pdf);;All files (*)")
        if not fname:
            return
        if not fname.lower().endswith(".pdf"):
            fname += ".pdf"
        self.figure.savefig(fname, dpi=300, bbox_inches="tight")
        QMessageBox.information(self, "Saved", f"Figure saved to {fname}")

    def _export_processed(self):
        if self._last_analysis is None:
            QMessageBox.information(self, "No data", "Run analysis first")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Export processed arrays (.npz)", "", "NPZ files (*.npz);;All files (*)")
        if not fname:
            return
        if not fname.lower().endswith(".npz"):
            fname += ".npz"
        res = self._last_analysis
        np.savez_compressed(
            fname,
            t_fs=res["t_fs"],
            tof=self.TOF,
            fAVG=res["fAVG"],
            fCNT=res["fCNT"],
            rfCNT=res["rfCNT"],
            S=res["S"],
            fit_params=res["p"],
        )
        QMessageBox.information(self, "Saved", f"Processed arrays saved to {fname}")

    def _on_plot_visibility_changed(self, state):
        if self._last_analysis is not None:
            self._create_or_update_artists(self._last_analysis)

    def run_analysis(self):
        if self._analysis_worker is not None and self._analysis_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Analysis is already running")
            return

        self.status.setText("Running analysis...")
        self.btn_run.setEnabled(False)

        t_fs = np.arange(self.spin_nstop.value() - self.spin_nstart.value())
        params = {
            "n_start": self.spin_nstart.value(),
            "n_stop": self.spin_nstop.value(),
            "n_pts": 1,
            "roi_min": self.spin_roi_min.value(),
            "roi_max": self.spin_roi_max.value(),
            "edge_level": 30,
            "model": self.model_combo.currentText(),
            "t_fs": t_fs,
            "fft": True,
            "tof_arr": self.TOF,
        }

        self._analysis_worker = AnalysisWorker(self.folder, self.data, params)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.start()

    def _on_analysis_finished(self, result):
        self.btn_run.setEnabled(True)
        if "error" in result:
            self.status.setText(f"Error: {result['error']}")
            QMessageBox.critical(self, "Analysis Error", result["error"])
            return
        self._last_analysis = result
        self.status.setText("Analysis complete")
        self._create_or_update_artists(result)

    # Pan/zoom
    def on_press(self, event):
        if event.button == 1 and event.inaxes:
            self._pan_state[event.inaxes] = {"x": event.xdata, "y": event.ydata}

    def on_motion(self, event):
        if event.button == 1 and event.inaxes in self._pan_state and event.xdata is not None:
            ax = event.inaxes
            dx = event.xdata - self._pan_state[ax]["x"]
            dy = event.ydata - self._pan_state[ax]["y"]
            xlim = ax.get_xlim()
            ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            if ax in self._current_im_axes:
                ylim = ax.get_ylim()
                ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.canvas.draw_idle()

    def on_release(self, event):
        if event.inaxes in self._pan_state:
            del self._pan_state[event.inaxes]

    def on_scroll(self, event):
        if not event.inaxes:
            return
        zoom = 1.1 if event.button == "up" else 1 / 1.1
        xlim = event.inaxes.get_xlim()
        event.inaxes.set_xlim(
            event.xdata - (event.xdata - xlim[0]) / zoom,
            event.xdata + (xlim[1] - event.xdata) / zoom,
        )
        if event.inaxes in self._current_im_axes:
            ylim = event.inaxes.get_ylim()
            event.inaxes.set_ylim(
                event.ydata - (event.ydata - ylim[0]) / zoom,
                event.ydata + (ylim[1] - event.ydata) / zoom,
            )
        self.canvas.draw_idle()

    def _create_or_update_artists(self, result):
        fCNT, rfCNT = result["fCNT"], result["rfCNT"]
        S, p, t_fs, fft = result["S"], result["p"], result["t_fs"], result["fft"]

        plot_data = {
            "Raw Avg": {"arr": -self.data["analog"], "xaxis": self.TOF},
            "Folded": {"arr": fCNT, "xaxis": self.TOF},
            "SC Corrected": {"arr": rfCNT, "xaxis": self.TOF},
            "FFT": {"x": fft["freq_ghz"] if fft else np.array([]), "y": fft["power"] if fft else np.array([])},
            "Dynamics Log": {"x": t_fs, "y": S},
            "Dynamics Lin": {"x": t_fs, "y": S},
            "Residuals": {"x": t_fs, "y": S - self.two_exp1(t_fs, *p) if p is not None and len(p) >= 7 else np.zeros_like(t_fs)},
        }

        if not self._artists_initialized:
            self.figure.clf()
            self._axes_list = []
            for r in range(3):
                for c in range(3):
                    ax = self.figure.add_subplot(3, 3, r * 3 + c + 1)
                    ax.tick_params(axis="both", which="major", labelsize=7)
                    self._axes_list.append(ax)

            for i, name in enumerate(self.ALL_PLOTS):
                ax = self._axes_list[i]
                if name in self.IMAGE_PLOTS:
                    im = ax.imshow([[0]], aspect="auto", origin="lower", cmap="viridis")
                    self._plot_artists[name] = {"ax": ax, "im": im}
                    self._current_im_axes[ax] = im
                else:
                    line, = ax.plot([], [], "-k", lw=1)
                    fit_line, = ax.plot([], [], "-r", lw=1.2)
                    self._plot_artists[name] = {"ax": ax, "line": line, "fit_line": fit_line}
                ax.set_title(name, fontsize=8)

            for j in range(len(self.ALL_PLOTS), 9):
                self._axes_list[j].set_visible(False)

            self.figure.tight_layout()
            self._artists_initialized = True

        for name in self.ALL_PLOTS:
            art = self._plot_artists.get(name)
            if art:
                art["ax"].set_visible(self.chk_plots[name].isChecked())

        for name in self.ALL_PLOTS:
            if not self.chk_plots[name].isChecked():
                continue
            art = self._plot_artists.get(name)
            if not art:
                continue
            ax = art["ax"]

            if name in self.IMAGE_PLOTS:
                im = art["im"]
                arr = plot_data[name]["arr"]
                xaxis = plot_data[name]["xaxis"]
                denom = float(np.abs(np.max(arr))) if arr.size else 1.0
                if denom == 0:
                    denom = 1.0
                im.set_data(arr / denom)
                im.set_extent([float(xaxis.min()), float(xaxis.max()), 0, arr.shape[0]])
                cfg = GLOBAL_SETTINGS["plots"].get(name, {})
                im.set_cmap(cfg.get("cmap", "viridis"))
                im.set_clim(cfg.get("vmin", 0.0), cfg.get("vmax", 0.4))
                ax.set_xlim(float(xaxis.min()), float(xaxis.max()))
                ax.set_ylim(0, arr.shape[0])
            else:
                line = art["line"]
                fit_line = art["fit_line"]
                xd = plot_data[name]["x"]
                yd = plot_data[name]["y"]

                if name == "Dynamics Log":
                    ax.set_xscale("log")
                    mask = xd > 0
                    xd = xd[mask]
                    yd = yd[mask]
                else:
                    ax.set_xscale("linear")

                line.set_data(xd, yd)

                if name in ["Dynamics Log", "Dynamics Lin"] and p is not None and len(p) >= 7:
                    try:
                        yfit = self.two_exp1(xd, *p)
                        fit_line.set_data(xd, yfit)
                        fit_line.set_visible(True)
                    except Exception:
                        fit_line.set_visible(False)
                else:
                    fit_line.set_visible(False)

                ax.relim()
                ax.autoscale_view()

            ax.set_title(name, fontsize=8)

        self.canvas.draw_idle()

class TOFExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TOF Explorer 2026")
        self.resize(1400, 900)

        self.data = None
        self.folder = None
        self._analysis_window = None

        self._updating = False
        self.cbar = None
        self._current_mesh = None
        self._current_mesh_shape = None
        self._current_x_centers = None

        self._last_axis_mode = None

        # Debounce timers
        self._calib_debounce_timer = QTimer(self)
        self._calib_debounce_timer.setSingleShot(True)
        self._calib_debounce_timer.timeout.connect(self._apply_view_calib)

        self._color_debounce_timer = QTimer(self)
        self._color_debounce_timer.setSingleShot(True)
        self._color_debounce_timer.timeout.connect(self._apply_color_limits)

        self._limit_debounce_timer = QTimer(self)
        self._limit_debounce_timer.setSingleShot(True)
        self._limit_debounce_timer.timeout.connect(self.update_plot)

        self.watch_timer = QTimer(self)
        self.watch_timer.setInterval(30_000)
        self.watch_timer.timeout.connect(self._poll_folder)
        self.last_file_list = []

        self._setup_ui()

        last = GLOBAL_SETTINGS.get("ui", {}).get("last_folder", "")
        if last and os.path.exists(last):
            self._start_loading(last)
            if GLOBAL_SETTINGS.get("ui", {}).get("auto_watch", False):
                self.watch_checkbox.setChecked(True)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.addLayout(self._create_left_panel(), 1)
        layout.addLayout(self._create_right_panel(), 4)

    def _create_left_panel(self):
        v = QVBoxLayout()

        self.btn_load = QPushButton("Load Folder")
        self.btn_load.clicked.connect(self._dlg_load)
        v.addWidget(self.btn_load)

        self.pbar = QProgressBar()
        v.addWidget(self.pbar)
        self.progress_label = QLabel("Idle")
        v.addWidget(self.progress_label)

        self.watch_checkbox = QCheckBox("Auto watch folder (30s)")
        self.watch_checkbox.stateChanged.connect(self._toggle_watch)
        v.addWidget(self.watch_checkbox)

        self.btn_analyze = QPushButton("Advanced Analysis")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.open_analysis)
        v.addWidget(self.btn_analyze)

        v.addSpacing(20)

        display_group = QGroupBox("Display Controls")
        dg = QVBoxLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Averaging (Analog)", "Counting"])
        self.mode_combo.currentIndexChanged.connect(self.update_plot)
        dg.addWidget(QLabel("Mode:"))
        dg.addWidget(self.mode_combo)

        self.chk_normalize = QCheckBox("Normalize by Row Mean")
        self.chk_normalize.stateChanged.connect(self.update_plot)
        dg.addWidget(self.chk_normalize)

        self.chk_calibrate = QCheckBox("Calibrate axis")
        self.chk_calibrate.stateChanged.connect(lambda *_: (self._axis_mode_changed(force=False), self.update_plot()))
        dg.addWidget(self.chk_calibrate)

        self.calib_combo = QComboBox()
        self.calib_combo.addItems(["TOF", "KE", "BE"])
        self.calib_combo.currentIndexChanged.connect(lambda *_: (self._axis_mode_changed(force=False), self.update_plot()))
        dg.addWidget(QLabel("Calibration:"))
        dg.addWidget(self.calib_combo)

        self.spin_eph = QDoubleSpinBox()
        self.spin_eph.setRange(0, 100)
        self.spin_eph.setDecimals(3)
        self.spin_eph.setValue(29.6)
        self.spin_eph.valueChanged.connect(lambda *_: (self._axis_mode_changed(force=False), self.update_plot()))
        dg.addWidget(QLabel("Photon E (eV):"))
        dg.addWidget(self.spin_eph)

        self.spin_view_tof_offset = QDoubleSpinBox()
        self.spin_view_tof_offset.setDecimals(3)
        self.spin_view_tof_offset.setRange(-1000, 1000)
        self.spin_view_tof_offset.setSingleStep(0.01)   # or 0.001 if you prefer
        self.spin_view_tof_offset.setValue(_safe_float(GLOBAL_SETTINGS["calibration"]["TOF_OFFSET_NS"]))
        self.spin_view_tof_offset.valueChanged.connect(self._on_view_calib_changed)
        dg.addWidget(QLabel("TOF offset (ns):"))
        dg.addWidget(self.spin_view_tof_offset)

        self.spin_view_workfunc = QDoubleSpinBox()
        self.spin_view_workfunc.setDecimals(3)
        self.spin_view_workfunc.setRange(-10, 10)
        self.spin_view_workfunc.setSingleStep(0.01)
        self.spin_view_workfunc.setValue(_safe_float(GLOBAL_SETTINGS["calibration"]["WORK_FUNCTION_EV"]))
        self.spin_view_workfunc.valueChanged.connect(self._on_view_calib_changed)
        dg.addWidget(QLabel("Work function (eV):"))
        dg.addWidget(self.spin_view_workfunc)

        self.cmap_combo = QComboBox()
        cmap_list = GLOBAL_SETTINGS["ui"]["colormaps"]
        self. cmap_combo.addItems(cmap_list)
        current_cmap = GLOBAL_SETTINGS["plots"].get("Raw Avg", {}).get("cmap", "viridis")
        if current_cmap in cmap_list:
            self.cmap_combo.setCurrentText(current_cmap)
        self.cmap_combo.currentIndexChanged.connect(self._on_cmap_changed)
        dg.addWidget(QLabel("Colormap:"))
        dg.addWidget(self.cmap_combo)
        
        display_group.setLayout(dg)
        v.addWidget(display_group)

        limits = QGroupBox("Plot Limits")
        lg = QGridLayout()

        self.spin_xmin = QDoubleSpinBox()
        self.spin_xmin.setDecimals(0)
        self.spin_xmin.setRange(-1e12, 1e12)
        self.spin_xmin.valueChanged.connect(self._limit_changed)

        self.spin_xmax = QDoubleSpinBox()
        self.spin_xmax.setDecimals(0)
        self.spin_xmax.setRange(-1e12, 1e12)
        self.spin_xmax.valueChanged.connect(self._limit_changed)

        self.spin_ymin = QSpinBox()
        self.spin_ymin.setRange(0, 100000)
        self.spin_ymin.valueChanged.connect(self._limit_changed)

        self.spin_ymax = QSpinBox()
        self.spin_ymax.setRange(0, 100000)
        self.spin_ymax.valueChanged.connect(self._limit_changed)

        self.spin_cmin = QDoubleSpinBox()
        self.spin_cmin.setDecimals(3)
        self.spin_cmin.setRange(-1e12, 1e12)
        self.spin_cmin.setSingleStep(0.01)
        self.spin_cmin.valueChanged.connect(self._on_color_limit_changed)
        
        self.spin_cmax = QDoubleSpinBox()
        self.spin_cmax.setDecimals(3)
        self.spin_cmax.setRange(-1e12, 1e12)
        self.spin_cmax.setSingleStep(0.01)
        self.spin_cmax.valueChanged.connect(self._on_color_limit_changed)

        lg.addWidget(QLabel("X min:"), 0, 0)
        lg.addWidget(self.spin_xmin, 0, 1)
        lg.addWidget(QLabel("X max:"), 1, 0)
        lg.addWidget(self.spin_xmax, 1, 1)
        lg.addWidget(QLabel("Y min:"), 2, 0)
        lg.addWidget(self.spin_ymin, 2, 1)
        lg.addWidget(QLabel("Y max:"), 3, 0)
        lg.addWidget(self.spin_ymax, 3, 1)
        lg.addWidget(QLabel("Color min:"), 4, 0)
        lg.addWidget(self.spin_cmin, 4, 1)
        lg.addWidget(QLabel("Color max:"), 5, 0)
        lg.addWidget(self.spin_cmax, 5, 1)

        limits.setLayout(lg)
        v.addWidget(limits)

        v.addStretch()
        return v

    def _create_right_panel(self):
        v = QVBoxLayout()
        self.figure = plt.figure(figsize=(10, 8))
        self.gs = GridSpec(2, 2, figure=self.figure, width_ratios=[8, 2], height_ratios=[2, 8], wspace=0.0, hspace=0.0)
        self.ax_hprof = self.figure.add_subplot(self.gs[0, 0])
        self.ax_main = self.figure.add_subplot(self.gs[1, 0], sharex=self.ax_hprof)
        self.ax_vprof = self.figure.add_subplot(self.gs[1, 1], sharey=self.ax_main)
        self.ax_cbar = self.figure.add_subplot(self.gs[0, 1])
        plt.setp(self.ax_hprof.get_xticklabels(), visible=False)
        plt.setp(self.ax_vprof.get_yticklabels(), visible=False)
        self.canvas = FigureCanvas(self.figure)
        v.addWidget(self.canvas)
        return v

    def _axis_mode(self):
        if self.chk_calibrate.isChecked():
            return self.calib_combo.currentText()
        return "TOF"

    def _compute_axis(self, tof):
        cal = CalibrationConstants()
        if self._axis_mode() == "KE":
            return tof_to_ke(
                tof, cal,
                bin_to_ns=bool(GLOBAL_SETTINGS["data"].get("BIN_TO_NS_FLAG", False)),
                points_per_ns=_safe_float(GLOBAL_SETTINGS["data"].get("POINTS_PER_NS", 1.0/0.8))
            )
        if self._axis_mode() == "BE":
            return tof_to_binding_energy(
                tof, _safe_float(self.spin_eph.value(), 29.6), cal,
                bin_to_ns=bool(GLOBAL_SETTINGS["data"].get("BIN_TO_NS_FLAG", False)),
                points_per_ns=_safe_float(GLOBAL_SETTINGS["data"].get("POINTS_PER_NS", 1.0/0.8))
            )
        return tof

    def _reset_x_limits_to_axis(self, axis):
        axis = np.asarray(axis, dtype=float)
        axis = axis[np.isfinite(axis)]
        if axis.size == 0:
            return
        xmin = float(np.min(axis))
        xmax = float(np.max(axis))
        if xmin == xmax:
            xmax = xmin + 1.0

        self.spin_xmin.blockSignals(True)
        self.spin_xmax.blockSignals(True)
        self.spin_xmin.setValue(xmin)
        self.spin_xmax.setValue(xmax)
        self.spin_xmin.blockSignals(False)
        self.spin_xmax.blockSignals(False)

    def _axis_mode_changed(self, force=False):
        if not self.data:
            return
        mode = self._axis_mode()
        if force or (mode != self._last_axis_mode):
            self._last_axis_mode = mode
            axis = self._compute_axis(self.data["tof"])
            self._reset_x_limits_to_axis(axis)

    def _on_color_limit_changed(self, val=None):
        self._color_debounce_timer.start(120)

    def _apply_color_limits(self):
        try:
            new_vmin = _safe_float(self.spin_cmin.value(), DEFAULT_SETTINGS["plots"]["Raw Avg"]["vmin"])
            new_vmax = _safe_float(self.spin_cmax.value(), DEFAULT_SETTINGS["plots"]["Raw Avg"]["vmax"])
            GLOBAL_SETTINGS["plots"].setdefault("Raw Avg", {})["vmin"] = new_vmin
            GLOBAL_SETTINGS["plots"].setdefault("Raw Avg", {})["vmax"] = new_vmax
            save_settings(GLOBAL_SETTINGS)

            mesh = getattr(self, "_current_mesh", None)
            if mesh is not None and hasattr(mesh, "set_clim"):
                mesh.set_clim(new_vmin, new_vmax)
                if self.cbar is not None:
                    try:
                        self.cbar.update_normal(mesh)
                    except Exception:
                        pass
                self.canvas.draw_idle()
            else:
                self.update_plot()
        except Exception:
            logger.exception("_apply_color_limits failed")

    def _limit_changed(self):
        if not self._updating:
            self._limit_debounce_timer.start(150)

    def _on_view_calib_changed(self):
        self._calib_debounce_timer.start(200)


    def _on_cmap_changed(self):
        """Handle colormap selection change"""
        new_cmap = self.cmap_combo.currentText()
        GLOBAL_SETTINGS["plots"]["Raw Avg"]["cmap"] = new_cmap
        save_settings(GLOBAL_SETTINGS)
        self.update_plot()

    def _apply_view_calib(self):
        GLOBAL_SETTINGS["calibration"]["TOF_OFFSET_NS"] = _safe_float(self.spin_view_tof_offset.value())
        GLOBAL_SETTINGS["calibration"]["WORK_FUNCTION_EV"] = _safe_float(self.spin_view_workfunc.value())
        save_settings(GLOBAL_SETTINGS)
        self._axis_mode_changed(force=True)
        self.update_plot()
        if self._analysis_window is not None:
            self._analysis_window.spin_tof_offset.setValue(_safe_float(GLOBAL_SETTINGS["calibration"]["TOF_OFFSET_NS"]))
            self._analysis_window.spin_workfunc.setValue(_safe_float(GLOBAL_SETTINGS["calibration"]["WORK_FUNCTION_EV"]))

    def _dlg_load(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self._start_loading(folder)

    def _start_loading(self, folder):
        GLOBAL_SETTINGS["ui"]["last_folder"] = folder
        save_settings(GLOBAL_SETTINGS)
        self.folder = folder
        self.pbar.setValue(0)
        self.progress_label.setText("Loading files...")
        self.btn_load.setEnabled(False)
        self.btn_analyze.setEnabled(False)

        self.loader = FastLoader(folder)
        self.loader.progress.connect(lambda p: self.pbar.setValue(int(p * 0.5)))
        self.loader.finished.connect(self._on_loaded)
        self.loader.start()

        self.last_file_list = sorted(glob.glob(os.path.join(folder, "TOF*.dat")))

    def _on_loaded(self, data):
        self.btn_load.setEnabled(True)
        if "error" in data:
            QMessageBox.critical(self, "Error", data["error"])
            self.pbar.setValue(0)
            self.progress_label.setText("Error")
            return

        self.data = data
        self.btn_analyze.setEnabled(True)

        self._init_spinboxes()
        self._axis_mode_changed(force=True)

        self.pbar.setValue(50)
        self.progress_label.setText("Loaded, rendering...")
        self.update_plot()

    def _init_spinboxes(self):
        for s in [self.spin_xmin, self.spin_xmax, self.spin_ymin, self.spin_ymax, self.spin_cmin, self.spin_cmax]:
            s.blockSignals(True)

        tof = self.data["tof"]
        n_files = self.data["analog"].shape[0]
        self.spin_xmin.setValue(float(np.nanmin(tof)))
        self.spin_xmax.setValue(float(np.nanmax(tof)))
        self.spin_ymin.setValue(0)
        self.spin_ymax.setValue(n_files)

        raw_vmin = GLOBAL_SETTINGS["plots"].get("Raw Avg", {}).get("vmin", DEFAULT_SETTINGS["plots"]["Raw Avg"]["vmin"])
        raw_vmax = GLOBAL_SETTINGS["plots"].get("Raw Avg", {}).get("vmax", DEFAULT_SETTINGS["plots"]["Raw Avg"]["vmax"])
        self.spin_cmin.setValue(_safe_float(raw_vmin, 0.0))
        self.spin_cmax.setValue(_safe_float(raw_vmax, 0.4))

        for s in [self.spin_xmin, self.spin_xmax, self.spin_ymin, self.spin_ymax, self.spin_cmin, self.spin_cmax]:
            s.blockSignals(False)

    def _toggle_watch(self, state):
        enabled = (state == Qt.Checked)
        GLOBAL_SETTINGS["ui"]["auto_watch"] = bool(enabled)
        save_settings(GLOBAL_SETTINGS)
        if enabled:
            self. watch_timer.start()
        else:
            self.watch_timer.stop()

    def _poll_folder(self):
        if not self.folder:
            return
        cur = sorted(glob. glob(os. path.join(self.folder, "TOF*. dat")))
        
        # Log every poll attempt
        logger. info(f"Auto-watch:  Checking folder (found {len(cur)} files)")
        
        # If no files exist, don't do anything
        if len(cur) == 0:
            logger.info("Auto-watch: No files in folder, skipping")
            return
        
        if cur != self.last_file_list:
            new_files = [f for f in cur if f not in self.last_file_list]
            removed_files = [f for f in self.last_file_list if f not in cur]
            
            if removed_files:
                logger.info(f"Files removed: {len(removed_files)}; full reload")
                self._start_loading(self.folder)
                self. last_file_list = cur
            elif new_files:
                logger.info(f"New files detected: {len(new_files)}; appending")
                logger.info(f"New files: {[os.path.basename(f) for f in new_files]}")
                self._append_new_files(new_files)
                self.last_file_list = cur
            else: 
                logger.info("Files changed; full reload")
                self._start_loading(self.folder)
                self.last_file_list = cur
        else:
            logger.info("Auto-watch: No changes detected")

    def _append_new_files(self, new_file_paths):
        """Incrementally load and append new TOF files to existing data"""
        if not self.data:
            self._start_loading(self.folder)
            return
        
        try:
            self.progress_label.setText(f"Loading {len(new_file_paths)} new files...")
            self.pbar.setValue(10)
            
            loader = FastLoader(self. folder)
            analog_list = []
            counting_list = []
            
            for i, fpath in enumerate(new_file_paths):
                try:
                    df = loader._read_tof_like_old(fpath)
                    if df.shape[1] < 2:
                        logger.warning(f"{fpath} has fewer than 2 columns — skipping")
                        continue
                    analog_list.append(df. iloc[:, 1]. to_numpy())
                    if df.shape[1] > 2:
                        counting_list.append(df.iloc[:, 2].to_numpy())
                    else:
                        counting_list.append(np.zeros_like(self.data["tof"]))
                except Exception as e:
                    logger.warning(f"Failed to parse {fpath}: {e} — skipping")
                    continue
                
                self.pbar.setValue(int(10 + 40 * (i + 1) / len(new_file_paths)))
            
            if not analog_list:
                logger.warning("No valid new files to append")
                self.progress_label.setText("Idle")
                self. pbar.setValue(100)
                return
            
            new_analog = np.vstack(analog_list)
            new_counting = np.vstack(counting_list)
            
            self. pbar.setValue(60)
            
            self.data["analog"] = np.vstack([self.data["analog"], new_analog])
            self.data["counting"] = np.vstack([self.data["counting"], new_counting])
            
            self.pbar.setValue(80)
            
            cache_file = os. path.join(self.folder, "processed_cache.npz")
            np.savez_compressed(
                cache_file,
                tof=self.data["tof"],
                analog=self.data["analog"],
                counting=self.data["counting"]
            )
            
            self. pbar.setValue(90)
            
            n_files = self.data["analog"].shape[0]
            self.spin_ymax.blockSignals(True)
            self.spin_ymax.setMaximum(n_files)
            self.spin_ymax.setValue(n_files)
            self.spin_ymax.blockSignals(False)
            
            self.update_plot()
            
            self. pbar.setValue(100)
            self.progress_label.setText(f"Idle (+{len(new_file_paths)} files)")
            logger.info(f"Successfully appended {len(new_file_paths)} new files")
            
        except Exception as e:
            logger.exception("Failed to append new files")
            self.progress_label.setText("Error appending files")
            QMessageBox.warning(self, "Append Error", f"Failed to append new files: {e}\n\nTry full reload.")

    def update_plot(self):
        if not self.data or self._updating:
            return
        self._updating = True
        try:
            self.pbar.setValue(85)
            self.progress_label.setText("Rendering viewer...")

            mode = self.mode_combo.currentIndex()
            intensity = self.data["analog"].copy() if mode == 0 else self.data["counting"].copy()
            tof = self.data["tof"]

            # old-app sign
            try:
                Sign = float(np.sign(intensity[0, np.argmax(np.abs(intensity[0, :]))]))
                if Sign == 0:
                    Sign = 1.0
            except Exception:
                Sign = 1.0
            intensity *= Sign

            if self.chk_normalize.isChecked():
                row_means = np.abs(np.mean(intensity, axis=1, keepdims=True))
                row_means[row_means == 0] = 1.0
                intensity = intensity / row_means

            axis = self._compute_axis(tof)

            xmin = _safe_float(self.spin_xmin.value(), float(np.nanmin(axis)))
            xmax = _safe_float(self.spin_xmax.value(), float(np.nanmax(axis)))
            xmin, xmax = (xmin, xmax) if xmin <= xmax else (xmax, xmin)

            ymin = int(self.spin_ymin.value())
            ymax = int(self.spin_ymax.value())
            ymin = max(0, ymin)
            ymax = min(intensity.shape[0], ymax)

            if ymin >= ymax:
                return

            idx_x = np.where((axis >= xmin) & (axis <= xmax))[0]
            if idx_x.size == 0:
                # if user limits exclude everything, fall back to full axis
                idx_x = np.arange(axis.size)

            x_full = axis[idx_x]
            sliced_data = intensity[ymin:ymax, :][:, idx_x]

            # normalize by max of displayed slice (old behavior)
            denom = float(np.abs(np.max(sliced_data))) if sliced_data.size else 1.0
            if denom == 0:
                denom = 1.0
            plotted = sliced_data / denom

            # downsample columns (optional performance)
            if plotted.shape[1] > MAX_DISPLAY_COLS:
                step = max(1, plotted.shape[1] // MAX_DISPLAY_COLS)
                plotted = plotted[:, ::step]
                x_full = x_full[::step]

            y_centers = np.arange(ymin, ymax)

            cmap_name = GLOBAL_SETTINGS["plots"].get("Raw Avg", {}).get("cmap", "viridis")
            cmin = _safe_float(GLOBAL_SETTINGS["plots"].get("Raw Avg", {}).get("vmin", 0.0), 0.0)
            cmax = _safe_float(GLOBAL_SETTINGS["plots"].get("Raw Avg", {}).get("vmax", 0.4), 0.4)

            # recreate mesh each time for correctness across axis mode changes
            self.ax_main.clear()
            self._current_mesh = self.ax_main.pcolormesh(x_full, y_centers, plotted, cmap=cmap_name, vmin=cmin, vmax=cmax, shading="auto")
            self.ax_main.set_xlim(float(np.min(x_full)), float(np.max(x_full)))
            self.ax_main.set_ylim(ymin, ymax)

            # profiles (old app uses mean on unnormalized? it used mean on Data; we use plotted to match display)
            # profiles MUST be computed from the same pre-display slice the old app used:
            #   SumH = mean(Data_slice, axis=0)
            #   SumV = mean(Data_slice, axis=1)
            # where Data_slice already includes sign correction + optional row-normalization.
            self.ax_hprof.clear()
            self.ax_vprof.clear()
            plt.setp(self.ax_hprof.get_xticklabels(), visible=False)
            plt.setp(self.ax_vprof.get_yticklabels(), visible=False)

            # IMPORTANT: if you downsampled plotted/x_full, mirror that for profiles too.
            # We want profiles aligned with x_full exactly.
            if plotted.shape[1] != sliced_data.shape[1]:
                # we downsampled by step; rebuild a matching downsample of sliced_data
                # (same step logic used above)
                step = max(1, sliced_data.shape[1] // MAX_DISPLAY_COLS)
                prof_data = sliced_data[:, ::step]
            else:
                prof_data = sliced_data

            # mean across rows -> horizontal profile vs x
            hprof = np.mean(prof_data, axis=0) if prof_data.size else np.array([])
            # mean across cols -> vertical profile vs y
            vprof = np.mean(prof_data, axis=1) if prof_data.size else np.array([])

            if hprof.size and x_full.size == hprof.size:
                self.ax_hprof.plot(x_full, hprof, "k-", lw=0.5)
                self.ax_hprof.set_xlim(float(np.min(x_full)), float(np.max(x_full)))

            if vprof.size:
                self.ax_vprof.plot(vprof, y_centers, "k-", lw=0.5)
                self.ax_vprof.set_ylim(ymin, ymax)
            xlabel = {"TOF": "TOF (ns)", "KE": "KE (eV)", "BE": "BE (eV)"}[self._axis_mode()]
            self.ax_main.set_xlabel(xlabel)
            self.ax_main.set_ylabel("File Index")

            # colorbar
            try:
                self.ax_cbar.cla()
                self.cbar = self.figure.colorbar(self._current_mesh, cax=self.ax_cbar)
            except Exception:
                pass

            self.canvas.draw_idle()
            self.pbar.setValue(100)
            self.progress_label.setText("Idle")
        finally:
            self._updating = False

    def open_analysis(self):
        if not self.data:
            QMessageBox.information(self, "No data", "Load a folder first")
            return
        if self._analysis_window and self._analysis_window.isVisible():
            self._analysis_window.activateWindow()
            self._analysis_window.raise_()
            return
        self._analysis_window = AnalysisWindow(self.folder, self.data, main_window=self)
        self._analysis_window.show()

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Delete Configuration?",
            "Do you want to delete the saved configuration file?\n\n"
            f"File: {CONFIG_PATH}",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.No,
        )
        if reply == QMessageBox.Cancel:
            event.ignore()
            return
        elif reply == QMessageBox.Yes:
            try:
                if os.path.exists(CONFIG_PATH):
                    os.remove(CONFIG_PATH)
                    logger.info(f"Configuration file deleted: {CONFIG_PATH}")
            except Exception as e:
                QMessageBox.warning(self, "Deletion Failed", str(e))
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = TOFExplorer()
    win.show()
    sys.exit(app.exec_())
