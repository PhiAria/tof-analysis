# ==============================================================================
# TOF ANALYSIS 2026 - DEVELOPMENT ROADMAP
# ==============================================================================
# ------------------------------------------------------------------------------
# PRIORITY 1
# ------------------------------------------------------------------------------
# [TODO-001] Plot Limits:  Update on Enter Key Press [X]
# [TODO-002] Energy Calibration:  Validation & Documentation [X]
# [TODO-003] Auto-Update:  Configurable Interval Selector []

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
        "TOF_OFFSET_NS":  16.0,  # t0 from notebook
        "WORK_FUNCTION_EV": 0.5,  # E0 from notebook (0.5*el / el = 0.5 eV)
        "FLIGHT_DISTANCE_M": 0.768,  # L from notebook (Johan's thesis)
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
        "t0_fixed_mm": 142.378,
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
    "analysis": {
        "file_start": 0,
        "file_stop": 2900,
        "model": "two_exp",
        "bin_to_ns": False,
        "edge_detection": {
            "tof_min_ns": 360.0,
            "tof_max_ns": 395.0,
            "level": 35
        },
        "fft": {
            "file_start": 0,
            "file_end": 100000,
            "tof_min_ns": 100.0,
            "tof_max_ns": 5600.0
        },
        "fitting": {
            "tof_min_ns": 370.0,
            "tof_max_ns": 400.0
        }
    }
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

def gaussian_profile(x, amp, center, sigma, offset):
    """Gaussian function for profile peak fitting"""
    return amp * np.exp(-(x - center)**2 / (2 * sigma**2)) + offset

def calculate_fwhm(sigma):
    """Calculate FWHM from Gaussian sigma"""
    return 2 * np.sqrt(2 * np.log(2)) * sigma


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


def load_log_file(folder_path):
    """
    Load log.dat file and extract scan parameters.
    Returns:  Npts, l_mm, t_fs, labtime
    """
    try:
        log_path = os.path.join(folder_path, "log.dat")
        if not os.path.exists(log_path):
            logger.warning(f"log.dat not found in {folder_path}, using defaults")
            return None, None, None, None
        
        LOG = np.loadtxt(log_path, usecols=(0, 1, 2))
        logger.info(f"Loaded log.dat with {LOG.shape[0]} lines")
        
        # Calculate number of points per scan
        Npts = int(np.max(LOG[:, 0]) // 2) + 1
        
        # Extract motor positions for first scan
        l_mm = LOG[0:Npts, 2]
        
        # Calculate time axis in femtoseconds
        l0_mm = GLOBAL_SETTINGS["fit"]. get("t0_fixed_mm", 142.298)
        speed_of_light = 299792458  # m/s
        t_fs = (l_mm - l0_mm) * 1e-3 / speed_of_light * 2 * 1e15
        
        # Calculate lab time
        labtime = LOG[:, 1] - LOG[0, 1]
        
        logger.info(f"Npts={Npts}, scan range: {np.min(t_fs):.1f} to {np.max(t_fs):.1f} fs")
        
        return Npts, l_mm, t_fs, labtime
        
    except Exception as e:
        logger.exception(f"Failed to load log.dat: {e}")
        return None, None, None, None


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
            # Check if binning is enabled
            BIN_TO_NS_FLAG = bool(GLOBAL_SETTINGS["data"]. get("BIN_TO_NS_FLAG", False))
            DATA_POINTS_PER_NS = int(_safe_float(GLOBAL_SETTINGS["data"].get("POINTS_PER_NS", 1.0/0.8)))
            
            for i, fpath in enumerate(files):
                try:
                    df = self._read_tof_like_old(fpath)
                    if df.shape[1] < 2:
                        logger.warning(f"{fpath} has fewer than 2 columns — skipping")
                        continue
                    
                    if tof_axis is None:
                        if BIN_TO_NS_FLAG and DATA_POINTS_PER_NS > 0:
                            # Bin TOF axis to ns
                            raw_tof = df.iloc[: , 0].to_numpy()
                            tof_axis = raw_tof[:: DATA_POINTS_PER_NS]
                        else:
                            tof_axis = df.iloc[:, 0].to_numpy()
                    
                    # Process analog signal
                    analog_raw = df.iloc[:, 1].to_numpy()
                    if BIN_TO_NS_FLAG and DATA_POINTS_PER_NS > 0:
                        # Reshape and average per ns
                        n_bins = len(analog_raw) // DATA_POINTS_PER_NS
                        if n_bins > 0:
                            _avg = np.reshape(analog_raw[: n_bins * DATA_POINTS_PER_NS], 
                                            (n_bins, DATA_POINTS_PER_NS))
                            analog_list.append(np.mean(_avg, axis=1))
                        else: 
                            analog_list.append(analog_raw)
                    else: 
                        analog_list.append(analog_raw)
                    
                    # Process counting signal
                    if df.shape[1] > 2:
                        counting_raw = df.iloc[:, 2]. to_numpy()
                        if BIN_TO_NS_FLAG and DATA_POINTS_PER_NS > 0:
                            # Reshape and sum per ns
                            n_bins = len(counting_raw) // DATA_POINTS_PER_NS
                            if n_bins > 0:
                                _cnt = np.reshape(counting_raw[: n_bins * DATA_POINTS_PER_NS],
                                                (n_bins, DATA_POINTS_PER_NS))
                                counting_list.append(np.sum(_cnt, axis=1))
                            else:
                                counting_list.append(counting_raw)
                        else:
                            counting_list.append(counting_raw)
                    else: 
                        # No counting data
                        if BIN_TO_NS_FLAG and DATA_POINTS_PER_NS > 0 and len(analog_list) > 0:
                            counting_list.append(np.zeros(len(analog_list[-1])))
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
            logger.info(f"Pump charge correction (a*tau/(b+tau)+c). a,b,c= {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}")
            return edges, fitted
        except Exception:
            return edges, np.zeros_like(tau)

    @staticmethod
    def correct_pump_charge(M, edge):
        """
        Correct pump-charge shift using np.roll (wraps at boundaries).
        Reference implementation from notebook.
        """
        rM = M.copy()
        for n in range(M.shape[0]):
            rM[n, :] = np.roll(M[n, :], int(edge[0] - edge[n]))
        return rM
    

    def run(self):
        try:
            AVG = self.data["analog"]
            CNT = self.data["counting"]
            n_start = int(self.params.get("n_start", 0))
            n_stop = int(self.params.get("n_stop", AVG.shape[0]))
            model_name = self.params.get("model", "two_exp")
            fft_requested = bool(self.params.get("fft", True))
            
            # NEW: Get BIN_TO_NS_FLAG and calculate POINTS_PER_NS
            bin_to_ns = bool(self.params.get("bin_to_ns", False))
            DATA_POINTS_PER_NS = 1.0 / 0.8  # Acqiris sampling rate (200ps)
            if bin_to_ns:
                POINTS_PER_NS = 1
            else:
                POINTS_PER_NS = DATA_POINTS_PER_NS
            
            logger.info("=== ANALYSIS STARTED ===")
            logger.info(f"Bin to 1ns: {bin_to_ns}")
            logger.info(f"POINTS_PER_NS: {POINTS_PER_NS}")
            
            # Load log.dat to get Npts and t_fs
            Npts, l_mm, t_fs, labtime = load_log_file(self.folder)
            
            # Fallback if log.dat not found
            if Npts is None:
                logger.warning("Using fallback: Npts=1, approximating t_fs")
                Npts = 1
                t_fs = np.arange(n_stop - n_start)
            else:
                logger.info(f"N points in the scan: {Npts}")
                logger.info(f"Scan range: {np.min(t_fs):.1f} fs to {np.max(t_fs):.1f} fs")
            
            # Log TOF file information
            n_tof_files = n_stop - n_start
            logger.info(f"N TOF files to read: {n_tof_files}")
            
            # Log data points info
            if CNT.shape[1] > 0:
                n_points_in_tof = CNT.shape[1]
                logger.info(f"N points within TOF files (after binning): {n_points_in_tof}")
            
            # Try to get N lines in TOF file (raw file info)
            try:
                tof_files = sorted(glob.glob(os.path.join(self.folder, "TOF*.dat")))
                if tof_files:
                    first_file = tof_files[n_start] if n_start < len(tof_files) else tof_files[0]
                    with open(first_file, 'r') as f:
                        n_lines = sum(1 for line in f if not line.startswith('#'))
                    logger.info(f"N lines in TOF file: {n_lines}")
            except Exception as e:
                logger.debug(f"Could not count lines in TOF file: {e}")
            
            self.progress.emit(0)
            
            # Fold CNT data (all processing on CNT)
            fAVG = self.fold_twoway(AVG[n_start:n_stop], Npts)
            fCNT = self.fold_twoway(CNT[n_start:n_stop], Npts)
            logger.info(f"Shape of the data matrix: {fCNT.shape}")
            self.progress.emit(15)

            # Edge detection range - now configurable
            edge_tof_min = float(self.params.get("edge_tof_min", 360))
            edge_tof_max = float(self.params.get("edge_tof_max", 395))
            edge_level = float(self.params.get("edge_level", 35))
            
            col0 = max(0, min(int(edge_tof_min * POINTS_PER_NS), fCNT.shape[1]-1))
            col1 = max(col0+1, min(int(edge_tof_max * POINTS_PER_NS), fCNT.shape[1]))
            
            logger.info(f"Edge detection TOF range: {edge_tof_min} ns to {edge_tof_max} ns")
            logger.info(f"Edge detection level: {edge_level}")
            
            edge_positions, fitted_edge = self.find_pump_charge(t_fs, fCNT[:, col0:col1], edge_level)
            self.progress.emit(35)

            # Use np.roll-based correction (matches reference notebook)
            rfCNT = self.correct_pump_charge(fCNT, edge_positions)
            rfAVG = self.correct_pump_charge(fAVG, edge_positions)
            self.progress.emit(55)

            # Get TOF range for fitting from parameters
            fit_tof_min = float(self.params.get("fit_tof_min", 370))
            fit_tof_max = float(self.params.get("fit_tof_max", 400))
            
            # Calculate TOF indices
            si1 = int(fit_tof_min * POINTS_PER_NS)
            si2 = int(fit_tof_max * POINTS_PER_NS)
            
            # Clamp to valid range
            si1 = max(0, min(si1, rfCNT.shape[1]-1))
            si2 = max(si1+1, min(si2, rfCNT.shape[1]))
            
            logger.info(f"Fitting TOF range: {fit_tof_min} ns to {fit_tof_max} ns")
            logger.info(f"TOF indices for fitting: {si1} to {si2}")
            
            # Sum over TOF range - ON CNT DATA
            S = np.sum(rfCNT[:, si1:si2], axis=1)
            self.progress.emit(65)

            # Initial guesses for fitting (UPDATED to match notebook)
            if model_name == "one_exp":
                p0 = [0, 30, 3000, 30, 10, 10]
            else:  # two_exp
                p0 = [0, 30, 1000, 30000, 100, 10, 0, 100]
            
            # ACTUAL CURVE FITTING
            p_full = None
            pcov = None
            perr = None
            fit_success = False
            
            try:
                if model_name == "one_exp":
                    fitfunc = AnalysisWindow.one_exp
                    # t0, sig, t1, A1, A3, B
                    lower_bounds = [-1000, 1, 1, 0, 0, -np.inf]
                    upper_bounds = [1000, 1000, 100000, np.inf, np.inf, np.inf]
                else:  # two_exp
                    fitfunc = AnalysisWindow.two_exp
                    # t0, sig, t1, t2, A1, A2, A3, B
                    lower_bounds = [-1000, 1, 1, 1, 0, 0, 0, -np.inf]
                    upper_bounds = [1000, 1000, 100000, 100000, np.inf, np.inf, np.inf, np.inf]
                
                # Perform the fit with bounds
                p_full, pcov = curve_fit(
                    fitfunc, 
                    t_fs, 
                    S, 
                    p0, 
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=10000
                )
                perr = np.sqrt(np.diag(pcov))
                fit_success = True
                
                logger.info(f"Fit successful with model: {model_name}")
                logger.info(f"Fitted parameters: {p_full}")
                logger.info(f"Parameter errors: {perr}")
                
            except Exception as e:
                logger.warning(f"Curve fitting failed: {e}, using initial guesses")
                p_full = np.asarray(p0, dtype=float)
                perr = np.zeros_like(p_full)
            
            self.progress.emit(80)

            # FFT on RAW CNT data (before folding, like notebook)
            fft_result = None
            if fft_requested:
                # Get FFT parameters
                fft_file_start = int(self.params.get("fft_file_start", 0))
                fft_file_end = int(self.params.get("fft_file_end", n_tof_files))
                fft_tof_min = float(self.params.get("fft_tof_min", 100))
                fft_tof_max = float(self.params.get("fft_tof_max", 5600))
                
                # Clamp file indices
                fft_file_start = max(0, min(fft_file_start, n_tof_files-1))
                fft_file_end = max(fft_file_start+1, min(fft_file_end, n_tof_files))
                
                # Calculate TOF indices for FFT
                fft_ind1 = max(0, min(int(fft_tof_min * POINTS_PER_NS), CNT.shape[1]-1))
                fft_ind2 = max(fft_ind1+1, min(int(fft_tof_max * POINTS_PER_NS), CNT.shape[1]))
                
                logger.info(f"FFT file range: {fft_file_start} to {fft_file_end}")
                logger.info(f"FFT TOF range: {fft_tof_min} ns to {fft_tof_max} ns")
                logger.info(f"FFT TOF indices: {fft_ind1} to {fft_ind2}")
                
                # Use RAW CNT data (not folded or corrected)
                raw_cnt_slice = CNT[n_start+fft_file_start:n_start+fft_file_end, fft_ind1:fft_ind2]
                time_series = np.sum(raw_cnt_slice, axis=1)
                
                logger.info(f"FFT using {len(time_series)} time points")
                
                # Perform FFT
                sig = time_series - np.mean(time_series)
                N = sig.size
                spec = np.fft.fft(sig)
                freq_bins = np.arange(N)
                power = np.abs(spec)
                fft_result = {"freq_bins": freq_bins, "power": power, "N": N}
                
                logger.info(f"FFT: N={N} points")
            
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
                "perr": perr,
                "fit_success": fit_success,
                "t_fs": t_fs,
                "fft": fft_result,
                "model_name": model_name,
                "Npts": len(t_fs) if hasattr(t_fs, '__len__') else 1,
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
        main_layout = QHBoxLayout(central)
        
        # LEFT PANEL:  controls
        controls = self._create_controls()
        main_layout.addLayout(controls, 1)
        
        # CENTER:  matplotlib canvas
        self.figure = plt.figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        main_layout.addWidget(self.canvas, 4)
        
        # RIGHT PANEL: plot customization
        customize_panel = self._create_customize_panel()
        main_layout.addLayout(customize_panel, 1)

    def _create_controls(self):
        v = QVBoxLayout()

        # ==================== Analysis Parameters ====================
        params = QGroupBox("Analysis parameters")
        p = QGridLayout()
        row = 0

        self.spin_l0 = QDoubleSpinBox()
        self.spin_l0.setRange(-1000, 1000)
        self.spin_l0.setDecimals(3)
        self.spin_l0.setValue(_safe_float(GLOBAL_SETTINGS["fit"].get("t0_fixed_mm", 142.378)))
        p.addWidget(QLabel("l0 (mm):"), row, 0)
        p.addWidget(self.spin_l0, row, 1)
        row += 1

        self.spin_nstart = QSpinBox()
        self.spin_nstart.setRange(0, 100000)
        self.spin_nstart.setValue(GLOBAL_SETTINGS["analysis"].get("file_start", 0))
        p.addWidget(QLabel("Start file index:"), row, 0)
        p.addWidget(self.spin_nstart, row, 1)
        row += 1

        self.spin_nstop = QSpinBox()
        self.spin_nstop.setRange(1, 100000)
        default_stop = GLOBAL_SETTINGS["analysis"].get("file_stop", 2900)
        self.spin_nstop.setValue(min(default_stop, self.data["analog"].shape[0]))
        p.addWidget(QLabel("Stop file index:"), row, 0)
        p.addWidget(self.spin_nstop, row, 1)
        row += 1

        self.model_combo = QComboBox()
        self.model_combo.addItems(["one_exp", "two_exp"])
        self.model_combo.setCurrentText(GLOBAL_SETTINGS["analysis"].get("model", "two_exp"))
        p.addWidget(QLabel("Model:"), row, 0)
        p.addWidget(self.model_combo, row, 1)
        row += 1

        # Bin to 1ns checkbox
        self.chk_bin_to_ns = QCheckBox("Bin to 1 ns")
        self.chk_bin_to_ns.setChecked(GLOBAL_SETTINGS["analysis"].get("bin_to_ns", False))

        params.setLayout(p)
        v.addWidget(params)

        # ==================== Edge Detection Parameters ====================
        # ==================== Edge Detection Parameters ====================
        edge_params = QGroupBox("Edge Detection")
        edge_layout = QGridLayout()
        edge_row = 0

        edge_cfg = GLOBAL_SETTINGS["analysis"]["edge_detection"]

        self.spin_edge_tof_min = QDoubleSpinBox()
        self.spin_edge_tof_min.setRange(0, 10000)
        self.spin_edge_tof_min.setDecimals(0)
        self.spin_edge_tof_min.setValue(edge_cfg.get("tof_min_ns", 360.0))
        edge_layout.addWidget(QLabel("TOF Min (ns):"), edge_row, 0)
        edge_layout.addWidget(self.spin_edge_tof_min, edge_row, 1)
        edge_row += 1

        self.spin_edge_tof_max = QDoubleSpinBox()
        self.spin_edge_tof_max.setRange(0, 10000)
        self.spin_edge_tof_max.setDecimals(0)
        self.spin_edge_tof_max.setValue(edge_cfg.get("tof_max_ns", 395.0))
        edge_layout.addWidget(QLabel("TOF Max (ns):"), edge_row, 0)
        edge_layout.addWidget(self.spin_edge_tof_max, edge_row, 1)
        edge_row += 1

        self.spin_edge_level = QSpinBox()
        self.spin_edge_level.setRange(0, 100)
        self.spin_edge_level.setValue(edge_cfg.get("level", 35))
        edge_layout.addWidget(QLabel("Level:"), edge_row, 0)
        edge_layout.addWidget(self.spin_edge_level, edge_row, 1)
        edge_row += 1

        edge_params.setLayout(edge_layout)
        v.addWidget(edge_params)
        # ==================== FFT Parameters ====================
        # ==================== FFT Parameters ====================
        fft_params = QGroupBox("FFT Parameters")
        fft_layout = QGridLayout()
        fft_row = 0

        fft_cfg = GLOBAL_SETTINGS["analysis"]["fft"]

        self.spin_fft_file_start = QSpinBox()
        self.spin_fft_file_start.setRange(0, 100000)
        self.spin_fft_file_start.setValue(fft_cfg.get("file_start", 0))
        fft_layout.addWidget(QLabel("File Index Start:"), fft_row, 0)
        fft_layout.addWidget(self.spin_fft_file_start, fft_row, 1)
        fft_row += 1

        self.spin_fft_file_end = QSpinBox()
        self.spin_fft_file_end.setRange(1, 100000)
        self.spin_fft_file_end.setValue(fft_cfg.get("file_end", 100000))
        fft_layout.addWidget(QLabel("File Index End:"), fft_row, 0)
        fft_layout.addWidget(self.spin_fft_file_end, fft_row, 1)
        fft_row += 1

        self.spin_fft_tof_min = QDoubleSpinBox()
        self.spin_fft_tof_min.setRange(0, 10000)
        self.spin_fft_tof_min.setDecimals(0)
        self.spin_fft_tof_min.setValue(fft_cfg.get("tof_min_ns", 100.0))
        fft_layout.addWidget(QLabel("TOF Min (ns):"), fft_row, 0)
        fft_layout.addWidget(self.spin_fft_tof_min, fft_row, 1)
        fft_row += 1

        self.spin_fft_tof_max = QDoubleSpinBox()
        self.spin_fft_tof_max.setRange(0, 10000)
        self.spin_fft_tof_max.setDecimals(0)
        self.spin_fft_tof_max.setValue(fft_cfg.get("tof_max_ns", 5600.0))
        fft_layout.addWidget(QLabel("TOF Max (ns):"), fft_row, 0)
        fft_layout.addWidget(self.spin_fft_tof_max, fft_row, 1)
        fft_row += 1

        fft_params.setLayout(fft_layout)
        v.addWidget(fft_params)

        # ==================== Fit Parameters ====================
        # ==================== Fit Parameters ====================
        fit_params = QGroupBox("Fit Parameters")
        fit_layout = QGridLayout()
        fit_row = 0

        fit_cfg = GLOBAL_SETTINGS["analysis"]["fitting"]

        self.spin_fit_tof_min = QDoubleSpinBox()
        self.spin_fit_tof_min.setRange(0, 10000)
        self.spin_fit_tof_min.setDecimals(0)
        self.spin_fit_tof_min.setValue(fit_cfg.get("tof_min_ns", 370.0))
        fit_layout.addWidget(QLabel("TOF Min (ns):"), fit_row, 0)
        fit_layout.addWidget(self.spin_fit_tof_min, fit_row, 1)
        fit_row += 1

        self.spin_fit_tof_max = QDoubleSpinBox()
        self.spin_fit_tof_max.setRange(0, 10000)
        self.spin_fit_tof_max.setDecimals(0)
        self.spin_fit_tof_max.setValue(fit_cfg.get("tof_max_ns", 400.0))
        fit_layout.addWidget(QLabel("TOF Max (ns):"), fit_row, 0)
        fit_layout.addWidget(self.spin_fit_tof_max, fit_row, 1)
        fit_row += 1

        fit_params.setLayout(fit_layout)
        v.addWidget(fit_params)
        # ==================== Show Plots ====================
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

        # ==================== Action Buttons ====================
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        v.addWidget(self.btn_run)

        self.btn_refit = QPushButton("Refit Only")
        self.btn_refit.clicked.connect(self.refit_only)
        self.btn_refit.setEnabled(False)  # Enabled after first analysis
        v.addWidget(self.btn_refit)

        self.btn_export_csv = QPushButton("Export Fit to CSV")
        self.btn_export_csv.clicked.connect(self._export_fit_to_csv)
        self.btn_export_csv.setEnabled(False)  # Enabled after successful fit
        v.addWidget(self.btn_export_csv)

        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        v.addStretch()
        return v
    
    def _create_customize_panel(self):
        """Create the right-side plot customization panel"""
        v = QVBoxLayout()
        
        customize = QGroupBox("Customize Selected Plot")
        cust_layout = QVBoxLayout()
        
        # Dropdown to select plot
        self.plot_select_combo = QComboBox()
        self.plot_select_combo.addItems(self.ALL_PLOTS)
        self.plot_select_combo.currentTextChanged.connect(self._on_plot_selected_for_custom)
        cust_layout.addWidget(QLabel("Select Plot:"))
        cust_layout.addWidget(self.plot_select_combo)
        
        # Colormap selector (for image plots only)
        self.cmap_combo_analysis = QComboBox()
        cmap_list = GLOBAL_SETTINGS["ui"]["colormaps"]
        self.cmap_combo_analysis.addItems(cmap_list)
        self.cmap_combo_analysis.currentTextChanged.connect(self._on_analysis_cmap_changed)
        cust_layout.addWidget(QLabel("Colormap:"))
        cust_layout.addWidget(self.cmap_combo_analysis)
        
        # Limits controls
        limits_grid = QGridLayout()
        
        self.spin_plot_xmin = QDoubleSpinBox()
        self.spin_plot_xmin.setDecimals(0)
        self.spin_plot_xmin.setRange(-1e12, 1e12)
        self.spin_plot_xmin.valueChanged.connect(self._on_plot_limit_changed)
        
        self.spin_plot_xmax = QDoubleSpinBox()
        self.spin_plot_xmax.setDecimals(0)
        self.spin_plot_xmax.setRange(-1e12, 1e12)
        self.spin_plot_xmax.valueChanged.connect(self._on_plot_limit_changed)
        
        self.spin_plot_ymin = QDoubleSpinBox()
        self.spin_plot_ymin.setDecimals(0)
        self.spin_plot_ymin.setRange(-1e12, 1e12)
        self.spin_plot_ymin.valueChanged.connect(self._on_plot_limit_changed)
        
        self.spin_plot_ymax = QDoubleSpinBox()
        self.spin_plot_ymax.setDecimals(0)
        self.spin_plot_ymax.setRange(-1e12, 1e12)
        self.spin_plot_ymax.valueChanged.connect(self._on_plot_limit_changed)
        
        self.spin_plot_cmin = QDoubleSpinBox()
        self.spin_plot_cmin.setDecimals(3)
        self.spin_plot_cmin.setRange(-1e12, 1e12)
        self.spin_plot_cmin.setSingleStep(0.01)
        self.spin_plot_cmin.valueChanged.connect(self._on_plot_color_changed)
        
        self.spin_plot_cmax = QDoubleSpinBox()
        self.spin_plot_cmax.setDecimals(3)
        self.spin_plot_cmax.setRange(-1e12, 1e12)
        self.spin_plot_cmax.setSingleStep(0.01)
        self.spin_plot_cmax.valueChanged.connect(self._on_plot_color_changed)
        
        limits_grid.addWidget(QLabel("X min:"), 0, 0)
        limits_grid.addWidget(self.spin_plot_xmin, 0, 1)
        limits_grid.addWidget(QLabel("X max:"), 1, 0)
        limits_grid.addWidget(self.spin_plot_xmax, 1, 1)
        limits_grid.addWidget(QLabel("Y min:"), 2, 0)
        limits_grid.addWidget(self.spin_plot_ymin, 2, 1)
        limits_grid.addWidget(QLabel("Y max:"), 3, 0)
        limits_grid.addWidget(self.spin_plot_ymax, 3, 1)
        limits_grid.addWidget(QLabel("Color min:"), 4, 0)
        limits_grid.addWidget(self.spin_plot_cmin, 4, 1)
        limits_grid.addWidget(QLabel("Color max:"), 5, 0)
        limits_grid.addWidget(self.spin_plot_cmax, 5, 1)
        
        cust_layout.addLayout(limits_grid)
        
        # Reset button
        self.btn_reset_plot = QPushButton("Reset to Auto")
        self.btn_reset_plot.clicked.connect(self._reset_plot_limits)
        cust_layout.addWidget(self.btn_reset_plot)
        
        customize.setLayout(cust_layout)
        v.addWidget(customize)
        v.addStretch()
        
        return v

    def _on_plot_visibility_changed(self, state):
        if self._last_analysis is not None:
            self._create_or_update_artists(self._last_analysis)
    
    def _on_plot_selected_for_custom(self, plot_name):
        """Update controls when user selects a plot to customize"""
        if not self._last_analysis or not plot_name: 
            return
        
        # Update spinbox values based on current plot settings
        cfg = GLOBAL_SETTINGS["plots"].get(plot_name, {})
        art = self._plot_artists.get(plot_name)
        
        if not art:
            return
        
        ax = art["ax"]
        
        # Block signals while updating
        for spin in [self.spin_plot_xmin, self.spin_plot_xmax, 
                     self.spin_plot_ymin, self.spin_plot_ymax,
                     self.spin_plot_cmin, self.spin_plot_cmax]:
            spin.blockSignals(True)
        
        # Set axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        self.spin_plot_xmin.setValue(xlim[0])
        self.spin_plot_xmax.setValue(xlim[1])
        self.spin_plot_ymin.setValue(ylim[0])
        self.spin_plot_ymax.setValue(ylim[1])
        
        # Set color limits and colormap for image plots
        if plot_name in self.IMAGE_PLOTS:
            self.cmap_combo_analysis.setEnabled(True)
            self.spin_plot_cmin.setEnabled(True)
            self.spin_plot_cmax.setEnabled(True)
            
            cmap_val = cfg.get("cmap", "viridis")
            self.cmap_combo_analysis.setCurrentText(cmap_val)
            
            vmin = cfg.get("vmin", 0.0)
            vmax = cfg.get("vmax", 0.4)
            self.spin_plot_cmin.setValue(vmin)
            self.spin_plot_cmax.setValue(vmax)
        else:
            self.cmap_combo_analysis.setEnabled(False)
            self.spin_plot_cmin.setEnabled(False)
            self.spin_plot_cmax.setEnabled(False)
        
        # Unblock signals
        for spin in [self.spin_plot_xmin, self.spin_plot_xmax, 
                     self.spin_plot_ymin, self.spin_plot_ymax,
                     self.spin_plot_cmin, self.spin_plot_cmax]:
            spin.blockSignals(False)
    
    def _on_analysis_cmap_changed(self):
        """Handle colormap change for selected plot"""
        plot_name = self.plot_select_combo.currentText()
        if not plot_name or plot_name not in self.IMAGE_PLOTS:
            return
        
        new_cmap = self.cmap_combo_analysis.currentText()
        GLOBAL_SETTINGS["plots"].setdefault(plot_name, {})["cmap"] = new_cmap
        save_settings(GLOBAL_SETTINGS)
        
        if self._last_analysis:
            self._update_single_plot(plot_name)
    
    def _on_plot_limit_changed(self):
        """Handle axis limit changes for selected plot"""
        plot_name = self.plot_select_combo.currentText()
        if not plot_name or not self._last_analysis:
            return
        
        art = self._plot_artists.get(plot_name)
        if not art:
            return
        
        ax = art["ax"]
        
        xmin = self.spin_plot_xmin.value()
        xmax = self.spin_plot_xmax.value()
        ymin = self.spin_plot_ymin.value()
        ymax = self.spin_plot_ymax.value()
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        self.canvas.draw_idle()
    
    def _on_plot_color_changed(self):
        """Handle color limit changes for selected image plot"""
        plot_name = self.plot_select_combo.currentText()
        if not plot_name or plot_name not in self.IMAGE_PLOTS:
            return
        
        vmin = self.spin_plot_cmin.value()
        vmax = self.spin_plot_cmax.value()
        
        GLOBAL_SETTINGS["plots"].setdefault(plot_name, {})["vmin"] = vmin
        GLOBAL_SETTINGS["plots"].setdefault(plot_name, {})["vmax"] = vmax
        save_settings(GLOBAL_SETTINGS)
        
        if self._last_analysis:
            self._update_single_plot(plot_name)
    
    def _reset_plot_limits(self):
        """Reset selected plot to automatic limits"""
        plot_name = self.plot_select_combo.currentText()
        if not plot_name or not self._last_analysis:
            return
        
        art = self._plot_artists.get(plot_name)
        if not art:
            return
        
        ax = art["ax"]
        ax.relim()
        ax.autoscale_view()
        
        # Update spinboxes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        self.spin_plot_xmin.blockSignals(True)
        self.spin_plot_xmax.blockSignals(True)
        self.spin_plot_ymin.blockSignals(True)
        self.spin_plot_ymax.blockSignals(True)
        
        self.spin_plot_xmin.setValue(xlim[0])
        self.spin_plot_xmax.setValue(xlim[1])
        self.spin_plot_ymin.setValue(ylim[0])
        self.spin_plot_ymax.setValue(ylim[1])
        
        self.spin_plot_xmin.blockSignals(False)
        self.spin_plot_xmax.blockSignals(False)
        self.spin_plot_ymin.blockSignals(False)
        self.spin_plot_ymax.blockSignals(False)
        
        self.canvas.draw_idle()
    
    def _update_single_plot(self, plot_name):
        """Refresh a single plot with updated settings"""
        if not self._last_analysis or not self._artists_initialized:
            return
        
        art = self._plot_artists.get(plot_name)
        if not art or not self.chk_plots[plot_name].isChecked():
            return
        
        # Re-run the plot update for this specific plot
        result = self._last_analysis
        fCNT, rfCNT = result["fCNT"], result["rfCNT"]
        S, p, t_fs, fft = result["S"], result["p"], result["t_fs"], result["fft"]
        model_name = result.get("model_name", "two_exp")
        
        # Calculate residuals based on the actual model
        residuals = np.zeros_like(t_fs)
        if p is not None and len(p) > 0:
            if model_name == "one_exp":
                fit_curve = self.one_exp(t_fs, *p)
            else:  # two_exp
                fit_curve = self.two_exp(t_fs, *p)
            residuals = S - fit_curve
        
        plot_data = {
            "Raw Avg": {"arr": -self.data["analog"], "xaxis": self.TOF},
            "Folded": {"arr": fCNT, "xaxis": self.TOF},
            "SC Corrected":  {"arr": rfCNT, "xaxis":  self.TOF},
            "FFT": {"x": fft["freq_bins"] if fft else np.array([]), "y": fft["power"] if fft else np.array([])},
            "Dynamics Log":  {"x": t_fs, "y": S},
            "Dynamics Lin": {"x": t_fs, "y": S},
            "Residuals": {"x": t_fs, "y": residuals},
        }
        
        ax = art["ax"]
        
        if plot_name in self.IMAGE_PLOTS:
            im = art["im"]
            arr = plot_data[plot_name]["arr"]
            xaxis = plot_data[plot_name]["xaxis"]
            denom = float(np.abs(np.max(arr))) if arr.size else 1.0
            if denom == 0:
                denom = 1.0
            im.set_data(arr / denom)
            
            cfg = GLOBAL_SETTINGS["plots"].get(plot_name, {})
            im.set_cmap(cfg.get("cmap", "viridis"))
            im.set_clim(cfg.get("vmin", 0.0), cfg.get("vmax", 0.4))
        
        self.canvas.draw_idle()

    def _create_expert_dock(self):
        """Create expert dock - placeholder for now"""
        pass

    def run_analysis(self):
        if self._analysis_worker is not None and self._analysis_worker.isRunning():
            logger.warning("Analysis already running")
            return

        self.status.setText("Running analysis...")
        self.btn_run.setEnabled(False)
        self.btn_refit.setEnabled(False)

        # Get BIN_TO_NS_FLAG from checkbox
        bin_to_ns = self.chk_bin_to_ns.isChecked()
        
        params = {
            "n_start": self.spin_nstart.value(),
            "n_stop": self.spin_nstop.value(),
            "model": self.model_combo.currentText(),
            "fft": True,
            "edge_tof_min": self.spin_edge_tof_min.value(),
            "edge_tof_max": self.spin_edge_tof_max.value(),
            "edge_level": self.spin_edge_level.value(),
            "bin_to_ns": bin_to_ns,
            "fft_file_start": self.spin_fft_file_start.value(),
            "fft_file_end": self.spin_fft_file_end.value(),
            "fft_tof_min": self.spin_fft_tof_min.value(),
            "fft_tof_max": self.spin_fft_tof_max.value(),
            "fit_tof_min": self.spin_fit_tof_min.value(),
            "fit_tof_max": self.spin_fit_tof_max.value(),
        }

        # Update ALL settings
        GLOBAL_SETTINGS["fit"]["t0_fixed_mm"] = self.spin_l0.value()
        GLOBAL_SETTINGS["analysis"]["file_start"] = self.spin_nstart.value()
        GLOBAL_SETTINGS["analysis"]["file_stop"] = self.spin_nstop.value()
        GLOBAL_SETTINGS["analysis"]["model"] = self.model_combo.currentText()
        GLOBAL_SETTINGS["analysis"]["bin_to_ns"] = bin_to_ns
        GLOBAL_SETTINGS["analysis"]["edge_detection"]["tof_min_ns"] = self.spin_edge_tof_min.value()
        GLOBAL_SETTINGS["analysis"]["edge_detection"]["tof_max_ns"] = self.spin_edge_tof_max.value()
        GLOBAL_SETTINGS["analysis"]["edge_detection"]["level"] = self.spin_edge_level.value()
        GLOBAL_SETTINGS["analysis"]["fft"]["file_start"] = self.spin_fft_file_start.value()
        GLOBAL_SETTINGS["analysis"]["fft"]["file_end"] = self.spin_fft_file_end.value()
        GLOBAL_SETTINGS["analysis"]["fft"]["tof_min_ns"] = self.spin_fft_tof_min.value()
        GLOBAL_SETTINGS["analysis"]["fft"]["tof_max_ns"] = self.spin_fft_tof_max.value()
        GLOBAL_SETTINGS["analysis"]["fitting"]["tof_min_ns"] = self.spin_fit_tof_min.value()
        GLOBAL_SETTINGS["analysis"]["fitting"]["tof_max_ns"] = self.spin_fit_tof_max.value()
        save_settings(GLOBAL_SETTINGS)

        self._analysis_worker = AnalysisWorker(self.folder, self.data, params)
        self._analysis_worker.progress.connect(lambda p: self.status.setText(f"Analysis: {p}%"))
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.start()
    
    def refit_only(self):
        """Re-run fitting only without recalculating everything"""
        if not self._last_analysis:
            logger.warning("No previous analysis to refit")
            return
        
        logger.info("=== REFIT ONLY ===")
        self.status.setText("Refitting...")
        self.btn_refit.setEnabled(False)
        
        try:
            result = self._last_analysis
            rfCNT = result["rfCNT"]
            t_fs = result["t_fs"]
            model_name = self.model_combo.currentText()
            
            # Get POINTS_PER_NS from checkbox state
            if self.chk_bin_to_ns.isChecked():
                POINTS_PER_NS = 1
            else:
                DATA_POINTS_PER_NS = 1.0 / 0.8
                POINTS_PER_NS = DATA_POINTS_PER_NS
            
            # Calculate TOF indices from ns values
            tof_min_ns = self.spin_fit_tof_min.value()
            tof_max_ns = self.spin_fit_tof_max.value()
            si1 = int(tof_min_ns * POINTS_PER_NS)
            si2 = int(tof_max_ns * POINTS_PER_NS)
            
            # Clamp to valid range
            si1 = max(0, min(si1, rfCNT.shape[1] - 1))
            si2 = max(si1 + 1, min(si2, rfCNT.shape[1]))
            
            # Sum over TOF range
            S = np.sum(rfCNT[:, si1:si2], axis=1)
            
            logger.info(f"Fitting TOF range: {tof_min_ns} ns to {tof_max_ns} ns")
            logger.info(f"TOF indices: {si1} to {si2}")
            logger.info(f"Sum signal over {si2-si1} TOF points")
            
            # Initial guesses and bounds based on model
            if model_name == "one_exp":
                p0 = [0, 30, 3000, 30, 10, 10]
                fitfunc = self.one_exp
                lower_bounds = [-1000, 1, 1, 0, 0, -np.inf]
                upper_bounds = [1000, 1000, 100000, np.inf, np.inf, np.inf]
            else:  # two_exp
                p0 = [0, 30, 1000, 30000, 100, 10, 0, 100]
                fitfunc = self.two_exp
                lower_bounds = [-1000, 1, 1, 1, 0, 0, 0, -np.inf]
                upper_bounds = [1000, 1000, 100000, 100000, np.inf, np.inf, np.inf, np.inf]
            
            # Perform fit with bounds
            p_full, pcov = curve_fit(
                fitfunc, 
                t_fs, 
                S, 
                p0, 
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000
            )
            perr = np.sqrt(np.diag(pcov))
            
            logger.info(f"Fit successful with model: {model_name}")
            
            # Log parameters with correct indexing
            if model_name == "one_exp":
                logger.info(f"  t0  = {p_full[0]:.6f} ± {perr[0]:.6f}")
                logger.info(f"  sig = {p_full[1]:.6f} ± {perr[1]:.6f}")
                logger.info(f"  t1  = {p_full[2]:.6f} ± {perr[2]:.6f}")
                logger.info(f"  A1  = {p_full[3]:.6f} ± {perr[3]:.6f}")
                logger.info(f"  A3  = {p_full[4]:.6f} ± {perr[4]:.6f}")
                logger.info(f"  B   = {p_full[5]:.6f} ± {perr[5]:.6f}")
            else:  # two_exp
                logger.info(f"  t0  = {p_full[0]:.6f} ± {perr[0]:.6f}")
                logger.info(f"  sig = {p_full[1]:.6f} ± {perr[1]:.6f}")
                logger.info(f"  t1  = {p_full[2]:.6f} ± {perr[2]:.6f}")
                logger.info(f"  t2  = {p_full[3]:.6f} ± {perr[3]:.6f}")
                logger.info(f"  A1  = {p_full[4]:.6f} ± {perr[4]:.6f}")
                logger.info(f"  A2  = {p_full[5]:.6f} ± {perr[5]:.6f}")
                logger.info(f"  A3  = {p_full[6]:.6f} ± {perr[6]:.6f}")
                logger.info(f"  B   = {p_full[7]:.6f} ± {perr[7]:.6f}")
            
            # Update result with new fit
            result["S"] = S
            result["p"] = p_full
            result["pcov"] = pcov
            result["perr"] = perr
            result["fit_success"] = True
            result["model_name"] = model_name
            
            # Only update dynamics plots
            self._update_dynamics_plots(result)
            
            self.status.setText("Refit complete")
            
        except Exception as e:
            logger.exception(f"Refit failed: {e}")
            self.status.setText(f"Refit error: {str(e)}")
        finally:
            self.btn_refit.setEnabled(True)

    def _update_dynamics_plots(self, result):
        """Update only the dynamics-related plots"""
        S = result["S"]
        p = result["p"]
        t_fs = result["t_fs"]
        model_name = result.get("model_name", "two_exp")
        
        # Get fit function
        if model_name == "one_exp":
            fitfunc = self.one_exp
        else:  # two_exp
            fitfunc = self.two_exp
        
        fit_curve = fitfunc(t_fs, *p)
        residuals = S - fit_curve
        
        # Update Dynamics Log
        if "Dynamics Log" in self._plot_artists and self.chk_plots["Dynamics Log"].isChecked():
            ax = self._plot_artists["Dynamics Log"]["ax"]
            ax.clear()
            ax.semilogy(t_fs, S, 'ko', ms=3, label='Data')
            ax.semilogy(t_fs, fit_curve, 'r-', lw=1, label='Fit')
            ax.set_title("Dynamics (Log)")
            ax.set_xlabel("Delay (fs)")
            ax.set_ylabel("Intensity")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Update Dynamics Lin
        if "Dynamics Lin" in self._plot_artists and self.chk_plots["Dynamics Lin"].isChecked():
            ax = self._plot_artists["Dynamics Lin"]["ax"]
            ax.clear()
            ax.plot(t_fs, S, 'ko', ms=3, label='Data')
            ax.plot(t_fs, fit_curve, 'r-', lw=1, label='Fit')
            ax.set_title("Dynamics (Linear)")
            ax.set_xlabel("Delay (fs)")
            ax.set_ylabel("Intensity")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Update Residuals
        if "Residuals" in self._plot_artists and self.chk_plots["Residuals"].isChecked():
            ax = self._plot_artists["Residuals"]["ax"]
            ax.clear()
            ax.plot(t_fs, residuals, 'ko', ms=2)
            ax.axhline(0, color='r', linestyle='--', lw=1)
            ax.set_title("Residuals")
            ax.set_xlabel("Delay (fs)")
            ax.set_ylabel("Data - Fit")
            ax.grid(True, alpha=0.3)
        
        self.canvas.draw_idle()



    def _export_fit_to_csv(self):
        """Export fit results and data to CSV file"""
        if not self._last_analysis:
            QMessageBox.warning(self, "No Data", "Run analysis first")
            return
        
        result = self._last_analysis
        
        # Get default filename
        folder_name = os.path.basename(self.folder) if self.folder else "analysis"
        default_filename = f"{folder_name}_fit_results.csv"
        
        # Open file dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Fit Results to CSV",
            default_filename,
            "CSV Files (*.csv)"
        )
        
        if not filename:
            return  # User cancelled
        
        try:
            import csv
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                writer.writerow(["# TOF Analysis Fit Results"])
                writer.writerow([f"# Folder: {self.folder}"])
                writer.writerow([f"# Model: {result.get('model_name', 'unknown')}"])
                writer.writerow([f"# Fit Success: {result.get('fit_success', False)}"])
                writer.writerow([])
                
                # Fit parameters
                writer.writerow(["# FIT PARAMETERS"])
                p = result.get("p")
                perr = result.get("perr")
                model_name = result.get("model_name", "two_exp")
                
                if p is not None and perr is not None:
                    writer.writerow(["Parameter", "Value", "Error", "Units"])
                    
                    if model_name == "one_exp":
                        writer.writerow(["t0", f"{p[0]:.6f}", f"{perr[0]:.6f}", "fs"])
                        writer.writerow(["sigma", f"{p[1]:.6f}", f"{perr[1]:.6f}", "fs"])
                        writer.writerow(["t1", f"{p[2]:.6f}", f"{perr[2]:.6f}", "fs"])
                        writer.writerow(["A1", f"{p[3]:.6f}", f"{perr[3]:.6f}", "a.u."])
                        writer.writerow(["A3", f"{p[4]:.6f}", f"{perr[4]:.6f}", "a.u."])
                        writer.writerow(["B", f"{p[5]:.6f}", f"{perr[5]:.6f}", "a.u."])
                    else:  # two_exp
                        writer.writerow(["t0", f"{p[0]:.6f}", f"{perr[0]:.6f}", "fs"])
                        writer.writerow(["sigma", f"{p[1]:.6f}", f"{perr[1]:.6f}", "fs"])
                        writer.writerow(["t1", f"{p[2]:.6f}", f"{perr[2]:.6f}", "fs"])
                        writer.writerow(["t2", f"{p[3]:.6f}", f"{perr[3]:.6f}", "fs"])
                        writer.writerow(["A1", f"{p[4]:.6f}", f"{perr[4]:.6f}", "a.u."])
                        writer.writerow(["A2", f"{p[5]:.6f}", f"{perr[5]:.6f}", "a.u."])
                        writer.writerow(["A3", f"{p[6]:.6f}", f"{perr[6]:.6f}", "a.u."])
                        writer.writerow(["B", f"{p[7]:.6f}", f"{perr[7]:.6f}", "a.u."])
                
                writer.writerow([])
                
                # Data: delay, signal, fit curve, residuals
                writer.writerow(["# DYNAMICS DATA"])
                writer.writerow(["Delay_fs", "Signal", "Fit", "Residuals"])
                
                t_fs = result.get("t_fs")
                S = result.get("S")
                
                if t_fs is not None and S is not None and p is not None:
                    # Calculate fit curve
                    if model_name == "one_exp":
                        fit_curve = self.one_exp(t_fs, *p)
                    else:
                        fit_curve = self.two_exp(t_fs, *p)
                    
                    residuals = S - fit_curve
                    
                    # Write data rows
                    for i in range(len(t_fs)):
                        writer.writerow([
                            f"{t_fs[i]:.6f}",
                            f"{S[i]:.6e}",
                            f"{fit_curve[i]:.6e}",
                            f"{residuals[i]:.6e}"
                        ])
            
            self.status.setText(f"Exported to {os.path.basename(filename)}")
            logger.info(f"Fit results exported to: {filename}")
            QMessageBox.information(self, "Export Successful", 
                                   f"Fit results saved to:\n{filename}")
            
        except Exception as e:
            logger.exception(f"CSV export failed: {e}")
            QMessageBox.critical(self, "Export Failed", 
                                f"Failed to export CSV:\n{str(e)}")


    def _on_analysis_finished(self, result):
        self.btn_run.setEnabled(True)
        if "error" in result:
            self.status.setText(f"Error: {result['error']}")
            QMessageBox.critical(self, "Analysis Error", result["error"])
            return

        self._last_analysis = result
        self.btn_refit.setEnabled(True)  # Enable refit button
        if result.get("fit_success"):
            self.btn_export_csv.setEnabled(True)  # Enable export if fit succeeded
        self.status.setText("Analysis complete")
        
        # Log fit results if available
        if result.get("fit_success") and result.get("perr") is not None:
            logger.info("=== FIT RESULTS ===")
            logger.info(f"Model: {result['model_name']}")
            p = result["p"]
            perr = result["perr"]
            model_name = result['model_name']
            
            if model_name == "one_exp":
                logger.info(f"  t0  = {p[0]:.6f} ± {perr[0]:.6f}")
                logger.info(f"  sig = {p[1]:.6f} ± {perr[1]:.6f}")
                logger.info(f"  t1  = {p[2]:.6f} ± {perr[2]:.6f}")
                logger.info(f"  A1  = {p[3]:.6f} ± {perr[3]:.6f}")
                logger.info(f"  A3  = {p[4]:.6f} ± {perr[4]:.6f}")
                logger.info(f"  B   = {p[5]:.6f} ± {perr[5]:.6f}")
            else:  # two_exp
                logger.info(f"  t0  = {p[0]:.6f} ± {perr[0]:.6f}")
                logger.info(f"  sig = {p[1]:.6f} ± {perr[1]:.6f}")
                logger.info(f"  t1  = {p[2]:.6f} ± {perr[2]:.6f}")
                logger.info(f"  t2  = {p[3]:.6f} ± {perr[3]:.6f}")
                logger.info(f"  A1  = {p[4]:.6f} ± {perr[4]:.6f}")
                logger.info(f"  A2  = {p[5]:.6f} ± {perr[5]:.6f}")
                logger.info(f"  A3  = {p[6]:.6f} ± {perr[6]:.6f}")
                logger.info(f"  B   = {p[7]:.6f} ± {perr[7]:.6f}")

        self._create_or_update_artists(result)

    def _create_or_update_artists(self, result):
        """Create or update all plot artists"""
        self.figure.clear()
        
        fAVG = result["fAVG"]
        fCNT = result["fCNT"]
        rfCNT = result["rfCNT"]
        rfAVG = result["rfAVG"]
        edge_positions = result["edge_positions"]
        fitted_edge = result.get("fitted_edge", np.array([]))
        S = result["S"]
        p = result["p"]
        t_fs = result["t_fs"]
        fft = result.get("fft")
        model_name = result.get("model_name", "two_exp")

        # Determine grid layout based on visible plots
        visible_plots = [name for name in self.ALL_PLOTS if self.chk_plots[name].isChecked()]
        n_plots = len(visible_plots)
        
        if n_plots == 0:
            return

        # Create grid
        nrows = (n_plots + 2) // 3
        ncols = min(n_plots, 3)
        
        self._plot_artists = {}
        self._axes_list = []
        
        for idx, plot_name in enumerate(visible_plots):
            ax = self.figure.add_subplot(nrows, ncols, idx + 1)
            self._axes_list.append(ax)
            
            cfg = GLOBAL_SETTINGS["plots"].get(plot_name, {})
            
            if plot_name == "Raw Avg": 
                arr = -self.data["analog"]
                denom = float(np.abs(np.max(arr))) if arr.size else 1.0
                if denom == 0:
                    denom = 1.0
                im = ax.imshow(arr / denom, aspect='auto', origin='lower',
                              extent=[self.TOF[0], self.TOF[-1], 0, arr.shape[0]],
                              cmap=cfg.get("cmap", "viridis"),
                              vmin=cfg.get("vmin", 0.0), vmax=cfg.get("vmax", 0.4))
                ax.set_title("Raw Avg")
                ax.set_xlabel("TOF (ns)")
                ax.set_ylabel("File Index")
                self.figure.colorbar(im, ax=ax)
                self._plot_artists[plot_name] = {"ax": ax, "im": im}
                
            elif plot_name == "Folded": 
                denom = float(np.abs(np.max(fCNT))) if fCNT.size else 1.0
                if denom == 0:
                    denom = 1.0
                im = ax.imshow(fCNT / denom, aspect='auto', origin='lower',
                              extent=[self.TOF[0], self.TOF[-1], t_fs[0], t_fs[-1]],
                              cmap=cfg.get("cmap", "viridis"),
                              vmin=cfg.get("vmin", 0.0), vmax=cfg.get("vmax", 0.4))
                ax.set_title("Folded")
                ax.set_xlabel("TOF (ns)")
                ax.set_ylabel("Delay (fs)")
                if edge_positions.size > 0:
                    ax.plot(edge_positions, t_fs[:len(edge_positions)], 'r.', ms=2, label='Edge')
                    if fitted_edge.size > 0:
                        ax.plot(fitted_edge[:len(t_fs)], t_fs, 'g-', lw=1, label='Fit')
                    ax.legend()
                self.figure.colorbar(im, ax=ax)
                self._plot_artists[plot_name] = {"ax": ax, "im": im}
                
            elif plot_name == "SC Corrected":
                denom = float(np.abs(np.max(rfCNT))) if rfCNT.size else 1.0
                if denom == 0:
                    denom = 1.0
                im = ax.imshow(rfCNT / denom, aspect='auto', origin='lower',
                              extent=[self.TOF[0], self.TOF[-1], t_fs[0], t_fs[-1]],
                              cmap=cfg.get("cmap", "viridis"),
                              vmin=cfg.get("vmin", 0.0), vmax=cfg.get("vmax", 0.4))
                ax.set_title("SC Corrected")
                ax.set_xlabel("TOF (ns)")
                ax.set_ylabel("Delay (fs)")
                self.figure.colorbar(im, ax=ax)
                self._plot_artists[plot_name] = {"ax": ax, "im": im}
                
            elif plot_name == "FFT":
                if fft: 
                    ax.semilogy(fft["freq_bins"], fft["power"], 'k-', lw=0.5)
                    ax.set_title("FFT")
                    ax.set_xlabel("Frequency bins")
                    ax.set_ylabel("Power")
                    ax.grid(True, alpha=0.3)
                self._plot_artists[plot_name] = {"ax": ax}
                
            elif plot_name == "Dynamics Log":
                ax.semilogy(t_fs, S, 'ko', ms=3, label='Data')
                if p is not None and len(p) > 0:
                    if model_name == "one_exp":
                        fit_curve = self.one_exp(t_fs, *p)
                    else:  # two_exp
                        fit_curve = self.two_exp(t_fs, *p)
                    ax.semilogy(t_fs, fit_curve, 'r-', lw=1, label='Fit')
                ax.set_title("Dynamics (Log)")
                ax.set_xlabel("Delay (fs)")
                ax.set_ylabel("Intensity")
                ax.legend()
                ax.grid(True, alpha=0.3)
                self._plot_artists[plot_name] = {"ax": ax}
                
            elif plot_name == "Dynamics Lin":
                ax.plot(t_fs, S, 'ko', ms=3, label='Data')
                if p is not None and len(p) > 0:
                    if model_name == "one_exp": 
                        fit_curve = self.one_exp(t_fs, *p)
                    else:  # two_exp
                        fit_curve = self.two_exp(t_fs, *p)
                    ax.plot(t_fs, fit_curve, 'r-', lw=1, label='Fit')
                ax.set_title("Dynamics (Linear)")
                ax.set_xlabel("Delay (fs)")
                ax.set_ylabel("Intensity")
                ax.legend()
                ax.grid(True, alpha=0.3)
                self._plot_artists[plot_name] = {"ax": ax}
                
            elif plot_name == "Residuals":
                if p is not None and len(p) > 0:
                    if model_name == "one_exp":
                        fit_curve = self.one_exp(t_fs, *p)
                    else:  # two_exp
                        fit_curve = self.two_exp(t_fs, *p)
                    residuals = S - fit_curve
                    ax.plot(t_fs, residuals, 'ko', ms=2)
                    ax.axhline(0, color='r', linestyle='--', lw=1)
                ax.set_title("Residuals")
                ax.set_xlabel("Delay (fs)")
                ax.set_ylabel("Data - Fit")
                ax.grid(True, alpha=0.3)
                self._plot_artists[plot_name] = {"ax": ax}

        self.figure.tight_layout()
        self._artists_initialized = True
        self.canvas.draw()

    def on_scroll(self, event):
        """Handle mouse scroll for zoom"""
        pass

    def on_press(self, event):
        """Handle mouse press for pan"""
        pass

    def on_motion(self, event):
        """Handle mouse motion for pan"""
        pass

    def on_release(self, event):
        """Handle mouse release"""
        pass


class BaselineWindow(QMainWindow):
    """Window for baseline selection and subtraction configuration"""
    
    def __init__(self, parent, baseline_folder, baseline_data):
        super().__init__()
        self.setWindowTitle(f"Baseline Subtraction: {os.path.basename(baseline_folder)}")
        self.resize(1200, 800)
        
        self.parent_window = parent
        self.baseline_folder = baseline_folder
        self.baseline_data = baseline_data
        self.baseline_tof = baseline_data["tof"]
        self.baseline_analog = baseline_data["analog"]
        self.baseline_counting = baseline_data["counting"]
        
        self._updating = False
        self.cbar = None
        self._current_mesh = None
        
        self._setup_ui()
        self._init_baseline_view()
        
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Left panel: controls
        layout.addLayout(self._create_controls(), 1)
        
        # Right panel: viewer
        layout.addLayout(self._create_viewer(), 4)
    
    def _create_controls(self):
        v = QVBoxLayout()
        
        # ROI Selection Group
        roi_group = QGroupBox("Baseline ROI (File Index)")
        roi_layout = QGridLayout()
        
        n_files = self.baseline_analog.shape[0]
        
        self.spin_file_start = QSpinBox()
        self.spin_file_start.setRange(0, max(0, n_files - 1))
        self.spin_file_start.setValue(0)
        self.spin_file_start.valueChanged.connect(self._update_baseline_view)
        
        self.spin_file_end = QSpinBox()
        self.spin_file_end.setRange(1, n_files)
        self.spin_file_end.setValue(n_files)
        self.spin_file_end.valueChanged.connect(self._update_baseline_view)
        
        roi_layout.addWidget(QLabel("File Start:"), 0, 0)
        roi_layout.addWidget(self.spin_file_start, 0, 1)
        roi_layout.addWidget(QLabel("File End:"), 1, 0)
        roi_layout.addWidget(self.spin_file_end, 1, 1)
        
        roi_group.setLayout(roi_layout)
        v.addWidget(roi_group)
        
        # Subtraction Mode Group
        mode_group = QGroupBox("Subtraction Mode")
        mode_layout = QVBoxLayout()
        
        self.chk_file_by_file = QCheckBox("File-by-file subtraction")
        self.chk_file_by_file.setChecked(False)
        self.chk_file_by_file.stateChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.chk_file_by_file)
        
        self.label_mode_info = QLabel("Mode: Total Average")
        self.label_mode_info.setWordWrap(True)
        mode_layout.addWidget(self.label_mode_info)
        
        mode_group.setLayout(mode_layout)
        v.addWidget(mode_group)
        
        # Display mode
        display_group = QGroupBox("Display")
        dg = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Averaging (Analog)", "Counting"])
        self.mode_combo.currentIndexChanged.connect(self._update_baseline_view)
        dg.addWidget(QLabel("Mode:"))
        dg.addWidget(self.mode_combo)
        
        display_group.setLayout(dg)
        v.addWidget(display_group)
        
        # Visualization limits (TOF axis - for display only)
        viz_group = QGroupBox("Visualization Limits (Display Only)")
        viz_layout = QGridLayout()
        
        self.spin_tof_min = QDoubleSpinBox()
        self.spin_tof_min.setDecimals(0)
        self.spin_tof_min.setRange(-1e12, 1e12)
        self.spin_tof_min.setValue(float(np.nanmin(self.baseline_tof)))
        self.spin_tof_min.valueChanged.connect(self._update_baseline_view)
        
        self.spin_tof_max = QDoubleSpinBox()
        self.spin_tof_max.setDecimals(0)
        self.spin_tof_max.setRange(-1e12, 1e12)
        self.spin_tof_max.setValue(float(np.nanmax(self.baseline_tof)))
        self.spin_tof_max.valueChanged.connect(self._update_baseline_view)
        
        viz_layout.addWidget(QLabel("TOF min:"), 0, 0)
        viz_layout.addWidget(self.spin_tof_min, 0, 1)
        viz_layout.addWidget(QLabel("TOF max:"), 1, 0)
        viz_layout.addWidget(self.spin_tof_max, 1, 1)
        
        viz_group.setLayout(viz_layout)
        v.addWidget(viz_group)
        
        v.addSpacing(20)
        
        # Action buttons
        self.btn_load_profiles = QPushButton("Load Profiles")
        self.btn_load_profiles.clicked.connect(self._load_profiles_only)
        v.addWidget(self.btn_load_profiles)
        
        self.btn_apply = QPushButton("Apply Subtraction")
        self.btn_apply.clicked.connect(self._apply_subtraction)
        v.addWidget(self.btn_apply)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel)
        v.addWidget(self.btn_cancel)
        
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        v.addWidget(self.status_label)
        
        v.addStretch()
        return v
    
    def _create_viewer(self):
        v = QVBoxLayout()
        self.figure = plt.figure(figsize=(10, 8))
        self.gs = GridSpec(2, 2, figure=self.figure, width_ratios=[8, 2], 
                          height_ratios=[2, 8], wspace=0.0, hspace=0.0)
        self.ax_hprof = self.figure.add_subplot(self.gs[0, 0])
        self.ax_main = self.figure.add_subplot(self.gs[1, 0], sharex=self.ax_hprof)
        self.ax_vprof = self.figure.add_subplot(self.gs[1, 1], sharey=self.ax_main)
        self.ax_cbar = self.figure.add_subplot(self.gs[0, 1])
        plt.setp(self.ax_hprof.get_xticklabels(), visible=False)
        plt.setp(self.ax_vprof.get_yticklabels(), visible=False)
        self.canvas = FigureCanvas(self.figure)
        v.addWidget(self.canvas)
        return v
    
    def _init_baseline_view(self):
        """Initialize the baseline view on window creation"""
        self._check_file_compatibility()
        self._update_baseline_view()
    
    def _check_file_compatibility(self):
        """Check if file-by-file mode is available"""
        parent_files = self.parent_window.data["analog"].shape[0] if self.parent_window.data else 0
        baseline_files = self.baseline_analog.shape[0]
        
        if baseline_files < parent_files:
            self.chk_file_by_file.setEnabled(False)
            self.chk_file_by_file.setChecked(False)
            self.status_label.setText(
                f"⚠️ File-by-file mode unavailable:\n"
                f"Baseline has {baseline_files} files, "
                f"main data has {parent_files} files.\n"
                f"Only averaged subtraction available."
            )
            logger.warning(
                f"Baseline has fewer files ({baseline_files}) than main data ({parent_files}). "
                f"File-by-file mode disabled."
            )
        else:
            self.chk_file_by_file.setEnabled(True)
            self.status_label.setText("Ready")
    
    def _on_mode_changed(self, state):
        """Handle subtraction mode change"""
        if self.chk_file_by_file.isChecked():
            if not self.chk_file_by_file.isEnabled():
                QMessageBox.warning(
                    self,
                    "Mode Unavailable",
                    "File-by-file subtraction is not available because the baseline "
                    "has fewer files than the main data.\n\n"
                    "Please use Total Average mode instead."
                )
                self.chk_file_by_file.setChecked(False)
                return
            self.label_mode_info.setText("Mode: File-by-file")
        else:
            self.label_mode_info.setText("Mode: Total Average")
    
    def _update_baseline_view(self):
        """Update the baseline visualization"""
        if self._updating:
            return
        
        self._updating = True
        try:
            mode = self.mode_combo.currentIndex()
            intensity = self.baseline_analog.copy() if mode == 0 else self.baseline_counting.copy()
            
            # Sign correction
            try:
                Sign = float(np.sign(intensity[0, np.argmax(np.abs(intensity[0, :]))]))
                if Sign == 0:
                    Sign = 1.0
            except Exception:
                Sign = 1.0
            intensity *= Sign
            
            # Get TOF limits for visualization
            tof_min = self.spin_tof_min.value()
            tof_max = self.spin_tof_max.value()
            
            # Get file index limits
            file_start = self.spin_file_start.value()
            file_end = self.spin_file_end.value()
            
            if file_start >= file_end:
                return
            
            # Filter by TOF
            idx_tof = np.where((self.baseline_tof >= tof_min) & (self.baseline_tof <= tof_max))[0]
            if idx_tof.size == 0:
                idx_tof = np.arange(self.baseline_tof.size)
            
            tof_filtered = self.baseline_tof[idx_tof]
            data_filtered = intensity[file_start:file_end, :][:, idx_tof]
            
            # Normalize
            denom = float(np.abs(np.max(data_filtered))) if data_filtered.size else 1.0
            if denom == 0:
                denom = 1.0
            plotted = data_filtered / denom

            # Downsample if needed
            if data_filtered.shape[1] > MAX_DISPLAY_COLS:
                step = max(1, data_filtered.shape[1] // MAX_DISPLAY_COLS)
                data_downsampled = data_filtered[:, ::step]
                tof_filtered = tof_filtered[::step]
            else:
                data_downsampled = data_filtered
            
            # Normalize AFTER determining what to plot
            denom = float(np.abs(np.max(data_downsampled))) if data_downsampled.size else 1.0
            if denom == 0:
                denom = 1.0
            plotted = data_downsampled / denom
            
            file_indices = np.arange(file_start, file_end)
            
            # Clear and plot
            self.ax_main.clear()
            self._current_mesh = self.ax_main.pcolormesh(
                tof_filtered, file_indices, plotted,
                cmap="viridis", vmin=0.0, vmax=0.4, shading="auto"
            )
            self.ax_main.set_xlim(float(np.min(tof_filtered)), float(np.max(tof_filtered)))
            self.ax_main.set_ylim(file_start, file_end)
            self.ax_main.set_xlabel("TOF (ns)")
            self.ax_main.set_ylabel("File Index")
            
            # Profiles - use unnormalized data
            self.ax_hprof.clear()
            self.ax_vprof.clear()
            plt.setp(self.ax_hprof.get_xticklabels(), visible=False)
            plt.setp(self.ax_vprof.get_yticklabels(), visible=False)
            
            hprof = np.mean(data_downsampled, axis=0) if data_downsampled.size else np.array([])
            vprof = np.mean(data_downsampled, axis=1) if data_downsampled.size else np.array([])
            
            # Plot profiles
            if hprof.size and tof_filtered.size == hprof.size:
                self.ax_hprof.plot(tof_filtered, hprof, "k-", lw=0.5)
                self.ax_hprof.set_xlim(float(np.min(tof_filtered)), float(np.max(tof_filtered)))
            
            if vprof.size:
                self.ax_vprof.plot(vprof, file_indices, "k-", lw=0.5)
                self.ax_vprof.set_ylim(file_start, file_end)
            
            # Colorbar
            try:
                self.ax_cbar.cla()
                self.cbar = self.figure.colorbar(self._current_mesh, cax=self.ax_cbar)
            except Exception:
                pass
            
            self.canvas.draw_idle()
            
        finally:
            self._updating = False
    
    def _apply_subtraction(self):
        """Apply baseline subtraction to parent data"""
        file_start = self.spin_file_start.value()
        file_end = self.spin_file_end.value()
        file_by_file = self.chk_file_by_file.isChecked()
        
        if file_start >= file_end:
            QMessageBox.warning(self, "Invalid ROI", "File Start must be less than File End")
            return
        
        # Prepare subtraction parameters
        subtraction_params = {
            "baseline_data": self.baseline_data,
            "file_start": file_start,
            "file_end": file_end,
            "file_by_file": file_by_file
        }
        
        # Call parent's subtraction method
        self.parent_window._apply_baseline_subtraction(subtraction_params)
        
        self.status_label.setText("✅ Subtraction applied!")



    def _load_profiles_only(self):
        """Load baseline profiles for display without applying subtraction"""
        file_start = self.spin_file_start.value()
        file_end = self.spin_file_end.value()
        
        if file_start >= file_end:
            QMessageBox.warning(self, "Invalid ROI", "File Start must be less than File End")
            return
        
        # Store baseline data AND the ROI used
        self.parent_window._baseline_data = self.baseline_data
        self.parent_window._baseline_roi = {  # ADD THESE 3 LINES
            "file_start": file_start,
            "file_end": file_end
        }
        
        # Preserve original data if not already preserved
        if self.parent_window._original_data is None:
            self.parent_window._original_data = {
                "analog": self.parent_window.data["analog"].copy(),
                "counting": self.parent_window.data["counting"].copy(),
                "tof": self.parent_window.data["tof"].copy()
            }
            logger.info("Original data preserved for baseline profile display")
        
        # Update main window plot to show profiles
        self.parent_window.update_plot()
        
        self.status_label.setText("✅ Baseline profiles loaded for display!")
        logger.info(f"Baseline profiles loaded from files {file_start}-{file_end} (no subtraction applied)")

    
    def _cancel(self):
        """Cancel and reset to original data"""
        self.parent_window._reset_baseline()
        self.close()




class TOFExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TOF Explorer 2026")
        # Auto-size to 80% of screen width, maintain 16:9 aspect ratio
        from PyQt5.QtWidgets import QDesktopWidget
        screen = QDesktopWidget().screenGeometry()
        width = int(screen.width() * 0.8)
        height = int(width * 9 / 16)  # 16:9 aspect ratio
        self.resize(width, height)

        self.data = None
        self.folder = None
        self._analysis_window = None
        self._baseline_window = None
        self._baseline_data = None
        self._original_data = None
        self._baseline_loader = None
        self._baseline_roi = None 
        
        self.coord_label = None  # Will be created in _create_right_panel

        self._updating = False
        # ... rest of __init__
        self.cbar = None
        self._current_mesh = None
        self._current_mesh_shape = None
        self._current_x_centers = None

        self._last_axis_mode = None
# Peak selection state
        self._peak_selection_active = False
        self._peak_selection_start = None
        self._peak_selection_rect = None
        self._peak_fit_lines = []
        self._peak_fit_text = None
        self._peak_current_profile = None  # 'horizontal' or 'vertical'
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
        layout.addLayout(self._create_left_panel(), 2)
        layout.addLayout(self._create_right_panel(), 5)

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
        self.btn_export_pdf = QPushButton("Export Plot as PDF")
        self.btn_export_pdf.setEnabled(False)  # Enable after data loads
        self.btn_export_pdf.clicked.connect(self._export_plot_to_pdf)
        v.addWidget(self.btn_export_pdf)

        self.btn_load_baseline = QPushButton("Load Baseline")
        self.btn_load_baseline.setEnabled(False)
        self.btn_load_baseline.clicked.connect(self._load_baseline)
        v.addWidget(self.btn_load_baseline)

        self.btn_reset_baseline = QPushButton("Reset Baseline Subtraction")
        self.btn_reset_baseline.setEnabled(False)
        self.btn_reset_baseline.clicked.connect(self._reset_baseline)
        v.addWidget(self.btn_reset_baseline)

        v.addSpacing(20)

        display_group = QGroupBox("Display Controls")
        dg = QVBoxLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Averaging (Analog)", "Counting"])
        self.mode_combo.currentIndexChanged.connect(self.update_plot)
        dg.addWidget(QLabel("Mode:"))
        dg.addWidget(self.mode_combo)



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
        self.spin_eph.setKeyboardTracking(False)  # Only update on Enter/focus loss
        self.spin_eph.editingFinished.connect(lambda: (self._axis_mode_changed(force=False), self.update_plot()))
        dg.addWidget(QLabel("Photon E (eV):"))
        dg.addWidget(self.spin_eph)

        self.spin_view_tof_offset = QDoubleSpinBox()
        self.spin_view_tof_offset.setDecimals(3)
        self.spin_view_tof_offset.setRange(-1000, 1000)
        self.spin_view_tof_offset.setSingleStep(0.01)
        self.spin_view_tof_offset.setValue(_safe_float(GLOBAL_SETTINGS["calibration"]["TOF_OFFSET_NS"]))
        self.spin_view_tof_offset.setKeyboardTracking(False)  # Only update on Enter/focus loss
        self.spin_view_tof_offset.editingFinished.connect(self._apply_view_calib)
        dg.addWidget(QLabel("TOF offset (ns):"))
        dg.addWidget(self.spin_view_tof_offset)

        self.spin_view_workfunc = QDoubleSpinBox()
        self.spin_view_workfunc.setDecimals(3)
        self.spin_view_workfunc.setRange(-10, 10)
        self.spin_view_workfunc.setSingleStep(0.01)
        self.spin_view_workfunc.setValue(_safe_float(GLOBAL_SETTINGS["calibration"]["WORK_FUNCTION_EV"]))
        self.spin_view_workfunc.setKeyboardTracking(False)  # Only update on Enter/focus loss
        self.spin_view_workfunc.editingFinished.connect(self._apply_view_calib)
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
        self.spin_xmin.setKeyboardTracking(False)  # Only update on Enter/focus loss
        self.spin_xmin.editingFinished.connect(self._apply_limits)

        self.spin_xmax = QDoubleSpinBox()
        self.spin_xmax.setDecimals(0)
        self.spin_xmax.setRange(-1e12, 1e12)
        self.spin_xmax.setKeyboardTracking(False)  # Only update on Enter/focus loss
        self.spin_xmax.editingFinished.connect(self._apply_limits)

        self.spin_ymin = QSpinBox()
        self.spin_ymin.setRange(0, 100000)
        self.spin_ymin.setKeyboardTracking(False)  # Only update on Enter/focus loss
        self.spin_ymin.editingFinished.connect(self._apply_limits)

        self.spin_ymax = QSpinBox()
        self.spin_ymax.setRange(0, 100000)
        self.spin_ymax.setKeyboardTracking(False)  # Only update on Enter/focus loss
        self.spin_ymax.editingFinished.connect(self._apply_limits)

        self.spin_cmin = QDoubleSpinBox()
        self.spin_cmin.setDecimals(3)
        self.spin_cmin.setRange(-1e12, 1e12)
        self.spin_cmin.setSingleStep(0.01)
        self.spin_cmin.setKeyboardTracking(False)  # Only update on Enter/focus loss
        self.spin_cmin.editingFinished.connect(self._apply_color_limits)

        self.spin_cmax = QDoubleSpinBox()
        self.spin_cmax.setDecimals(3)
        self.spin_cmax.setRange(-1e12, 1e12)
        self.spin_cmax.setSingleStep(0.01)
        self.spin_cmax.setKeyboardTracking(False)  # Only update on Enter/focus loss
        self.spin_cmax.editingFinished.connect(self._apply_color_limits)

        
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

        # ==================== Peak Analysis ====================
        peak_group = QGroupBox("Peak Analysis (Profiles)")
        peak_layout = QVBoxLayout()

        peak_info = QLabel("Enable selection, then click and drag\non horizontal or vertical profile to fit peak.")
        peak_info.setWordWrap(True)
        peak_info.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        peak_layout.addWidget(peak_info)

        self.btn_enable_peak = QPushButton("Enable Peak Selection")
        self.btn_enable_peak.setCheckable(True)
        self.btn_enable_peak.setEnabled(False)  # Enabled after data load
        self.btn_enable_peak.clicked.connect(self._toggle_peak_selection)
        peak_layout.addWidget(self.btn_enable_peak)


        self.peak_result_label = QLabel("FWHM: ---")
        self.peak_result_label.setStyleSheet("QLabel { font-family: monospace; padding: 5px; background-color: #f0f0f0; border: 1px solid #d0d0d0; }")
        self.peak_result_label.setWordWrap(True)
        peak_layout.addWidget(self.peak_result_label)

        self.btn_clear_peaks = QPushButton("Clear Peak Fits")
        self.btn_clear_peaks.clicked.connect(self._clear_peak_fits)
        self.btn_clear_peaks.setEnabled(False)
        peak_layout.addWidget(self.btn_clear_peaks)


        peak_group.setLayout(peak_layout)
        v.addWidget(peak_group)

        v.addStretch()
        return v


    def _create_right_panel(self):
        v = QVBoxLayout()
        
        # Add coordinate tracker at the top - compact version
        self.coord_label = QLabel("X: --- | Y: ---")
        self.coord_label.setStyleSheet(
            "QLabel { "
            "font-family: monospace; "
            "padding: 2px 5px; "
            "background-color: #f0f0f0; "
            "border: 1px solid #d0d0d0; "
            "}"
        )
        self.coord_label.setMaximumHeight(25)  # Limit height to one line
        v.addWidget(self.coord_label)
        
        self.figure = plt.figure(figsize=(10, 8))
        self.gs = GridSpec(2, 2, figure=self.figure, width_ratios=[8, 2], height_ratios=[2, 8], wspace=0.0, hspace=0.0)
        self.ax_hprof = self.figure.add_subplot(self.gs[0, 0])
        self.ax_main = self.figure.add_subplot(self.gs[1, 0], sharex=self.ax_hprof)
        self.ax_vprof = self.figure.add_subplot(self.gs[1, 1], sharey=self.ax_main)
        self.ax_cbar = self.figure.add_subplot(self.gs[0, 1])
        plt.setp(self.ax_hprof.get_xticklabels(), visible=False)
        plt.setp(self.ax_vprof.get_yticklabels(), visible=False)
        self.canvas = FigureCanvas(self.figure)
        
        # Connect mouse motion event
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self.canvas.mpl_connect('button_release_event', self._on_canvas_release)
        
        v.addWidget(self.canvas)
        return v
        
        self.figure = plt.figure(figsize=(10, 8))
        self.gs = GridSpec(2, 2, figure=self.figure, width_ratios=[8, 2], height_ratios=[2, 8], wspace=0.0, hspace=0.0)
        self.ax_hprof = self.figure.add_subplot(self.gs[0, 0])
        self.ax_main = self.figure.add_subplot(self.gs[1, 0], sharex=self.ax_hprof)
        self.ax_vprof = self.figure.add_subplot(self.gs[1, 1], sharey=self.ax_main)
        self.ax_cbar = self.figure.add_subplot(self.gs[0, 1])
        plt.setp(self.ax_hprof.get_xticklabels(), visible=False)
        plt.setp(self.ax_vprof.get_yticklabels(), visible=False)
        self.canvas = FigureCanvas(self.figure)
        
        # Connect mouse motion event
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        
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

    def _apply_limits(self):
        """Apply limit changes immediately (called on Enter key or focus loss)"""
        if not self._updating:
            self.update_plot()




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

    

    def _on_loaded(self, data):
        self.btn_load.setEnabled(True)
        if "error" in data:
            QMessageBox.critical(self, "Error", data["error"])
            self.pbar.setValue(0)
            self.progress_label.setText("Error")
            return

        self.data = data
        self.btn_analyze.setEnabled(True)
        self.btn_export_pdf.setEnabled(True)
        self.btn_load_baseline.setEnabled(True)
        self.btn_enable_peak.setEnabled(True)  # Enable peak selection

        self._init_spinboxes()
        self._axis_mode_changed(force=True)

        self.pbar.setValue(50)
        self.progress_label.setText("Loaded, rendering...")
        self.update_plot()

        if self.folder:
            current_files_on_disk = sorted(glob.glob(os.path.join(self.folder, "TOF*.dat")))
            n_files_in_memory = self.data["analog"].shape[0]
            
            if data.get("cached"):
                # Loaded from cache - might have fewer files than on disk
                # Set last_file_list to match what's in memory (first N files)
                self.last_file_list = current_files_on_disk[:n_files_in_memory]
                logger.info(f"Loaded from cache: {n_files_in_memory} files. Watch list initialized.")
                
                # Check if there are newer files
                if len(current_files_on_disk) > n_files_in_memory:
                    logger.info(f"Found {len(current_files_on_disk) - n_files_in_memory} newer files on disk.")
                    logger.info("They will be detected on next auto-watch poll.")
            else:
                # Fresh load - should match exactly
                self.last_file_list = current_files_on_disk
                logger.info(f"Fresh load: {n_files_in_memory} files loaded and tracked.")


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
        
        # Convert to sets for faster comparison
        cur_set = set(cur)
        last_set = set(self.last_file_list)

        if cur_set != last_set:
            new_files = list(cur_set - last_set)  # Files in cur but not in last
            removed_files = list(last_set - cur_set)  # Files in last but not in cur
    
            if removed_files:
                logger.info(f"Files removed: {len(removed_files)}; full reload required")
                self._start_loading(self.folder)
                self.last_file_list = cur
            elif new_files:
        # Sort new files numerically for proper order
                new_files_sorted = sorted(new_files, key=lambda f: self._extract_file_number(f))
                logger.info(f"New files detected: {len(new_files_sorted)}")
                logger.info(f"New files: {[os.path.basename(f) for f in new_files_sorted[:5]]}{'...' if len(new_files_sorted) > 5 else ''}")
                self._append_new_files(new_files_sorted)
                self.last_file_list = cur
            else:
        # This shouldn't happen, but just in case
                logger.warning("File list changed but no new/removed files detected; full reload")
                self._start_loading(self.folder)
                self.last_file_list = cur
        else:
            logger.debug("Auto-watch: No changes detected")

# Add helper method if not present
def _extract_file_number(self, filepath):
    """Extract numeric file index from TOF filename"""
    import re
    base = os.path.basename(filepath)
    m = re.search(r'(\d+)', base)
    return int(m.group(1)) if m else 0
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
            
# Store current ROI before updating
            old_ymin = self.spin_ymin.value()
            old_ymax = self.spin_ymax.value()
            old_xmin = self.spin_xmin.value()
            old_xmax = self.spin_xmax.value()

            n_files = self.data["analog"].shape[0]
            self.spin_ymax.blockSignals(True)
            self.spin_ymin.blockSignals(True)
            self.spin_xmin.blockSignals(True)
            self.spin_xmax.blockSignals(True)

# Update maximum allowed range
            self.spin_ymax.setMaximum(n_files)

# Restore previous ROI if it's still valid
            if old_ymax <= n_files:
    # Old ROI still fits - keep it
                self.spin_ymin.setValue(old_ymin)
                self.spin_ymax.setValue(old_ymax)
                logger.info(f"Maintained ROI: Y={old_ymin}-{old_ymax}")
            else:
    # Old ROI exceeds new data - extend to include new files
                self.spin_ymin.setValue(old_ymin)
                self.spin_ymax.setValue(n_files)
                logger.info(f"Extended ROI to include new files: Y={old_ymin}-{n_files}")

# Restore X limits
            self.spin_xmin.setValue(old_xmin)
            self.spin_xmax.setValue(old_xmax)

            self.spin_ymax.blockSignals(False)
            self.spin_ymin.blockSignals(False)
            self.spin_xmin.blockSignals(False)
            self.spin_xmax.blockSignals(False)

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
            if mode == 0:  # AVG
                sign = -1
            if mode == 1:  # CNT
                sign = 1
            intensity = intensity * sign



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
                # Plot current data profile
                self.ax_hprof.plot(x_full, hprof, "k-", lw=0.5, label='Current')
                
                # If baseline subtraction is active, show original (red) and baseline (blue)
                if self._original_data is not None and self._baseline_data is not None:
                    # Original data profile (use absolute values for display)
                    orig_intensity = self._original_data["analog"].copy() if mode == 0 else self._original_data["counting"].copy()
                    orig_intensity = np.abs(orig_intensity)  # Use absolute values
                    orig_sliced = orig_intensity[ymin:ymax, :][:, idx_x]
                    if orig_sliced.shape[1] != prof_data.shape[1]:
                        step = max(1, sliced_data.shape[1] // MAX_DISPLAY_COLS)
                        orig_prof_data = orig_sliced[:, ::step]
                    else:
                        orig_prof_data = orig_sliced
                    orig_hprof = np.mean(orig_prof_data, axis=0) if orig_prof_data.size else np.array([])
                    if orig_hprof.size:
                        self.ax_hprof.plot(x_full, orig_hprof, "r-", lw=0.5, alpha=0.7, label='Original')
                    
                    # Baseline profile (use absolute values for display)
                    base_intensity = self._baseline_data["analog"].copy() if mode == 0 else self._baseline_data["counting"].copy()
                    base_intensity = np.abs(base_intensity)  # Use absolute values
                    
                    # Use baseline ROI if specified, otherwise use current y-limits
                    if self._baseline_roi is not None:
                        roi_start = self._baseline_roi["file_start"]
                        roi_end = self._baseline_roi["file_end"]
                    else:
                        roi_start = min(ymin, base_intensity.shape[0]-1)
                        roi_end = min(ymax, base_intensity.shape[0])
                    
                    if base_intensity.shape[0] > 0:
                        base_sliced = base_intensity[roi_start:roi_end, :][:, idx_x]
                        if base_sliced.shape[1] != prof_data.shape[1]:
                            step = max(1, sliced_data.shape[1] // MAX_DISPLAY_COLS)
                            base_prof_data = base_sliced[:, ::step]
                        else:
                            base_prof_data = base_sliced
                        base_hprof = np.mean(base_prof_data, axis=0) if base_prof_data.size else np.array([])
                        if base_hprof.size:
                            self.ax_hprof.plot(x_full, base_hprof, "b-", lw=0.5, alpha=0.7, label='Baseline')
                    
                    self.ax_hprof.legend(loc='upper right', fontsize=8)
                
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

    def _toggle_peak_selection(self, checked):
        """Toggle peak selection mode"""
        self._peak_selection_active = checked
        if checked:
            self.btn_enable_peak.setText("Disable Peak Selection")
            self.progress_label.setText("Peak mode: drag on profile")
        else:
            self.btn_enable_peak.setText("Enable Peak Selection")
            self.progress_label.setText("Idle")

    def _clear_peak_fits(self):
        """Clear all peak fits"""
        for line in self._peak_fit_lines:
            try:
                line.remove()
            except:
                pass
        self._peak_fit_lines = []
    
        if self._peak_fit_text is not None:
            try:
                self._peak_fit_text.remove()
            except:
                pass
            self._peak_fit_text = None
    
        self.peak_result_label.setText("FWHM: ---")
        self.btn_clear_peaks.setEnabled(False)
        self.canvas.draw_idle()

    def _on_canvas_click(self, event):
        """Handle mouse button press for peak selection"""
        if not self._peak_selection_active or event.button != 1:
            return
    
    # Determine which profile was clicked
        if event.inaxes == self.ax_hprof:
            self._peak_selection_start = event.xdata
            self._peak_current_profile = 'horizontal'
            logger.info(f"Peak selection started on horizontal profile at x={event.xdata:.2f}")
        elif event.inaxes == self.ax_vprof:
            self._peak_selection_start = event.ydata
            self._peak_current_profile = 'vertical'
            logger.info(f"Peak selection started on vertical profile at y={event.ydata:.2f}")

    def _on_canvas_release(self, event):
        """Handle mouse button release for peak selection"""
        if not self._peak_selection_active or event.button != 1:
            return
    
        if self._peak_selection_start is None or self._peak_current_profile is None:
            return
    
    # Get end coordinate
        if self._peak_current_profile == 'horizontal' and event.inaxes == self.ax_hprof:
            if event.xdata is None:
                self._peak_selection_start = None
                self._peak_current_profile = None
                return
        
            x_min = min(self._peak_selection_start, event.xdata)
            x_max = max(self._peak_selection_start, event.xdata)
        
            logger.info(f"Horizontal profile selection: {x_min:.2f} to {x_max:.2f}")
            self._fit_horizontal_peak(x_min, x_max)
        
        elif self._peak_current_profile == 'vertical' and event.inaxes == self.ax_vprof:
            if event.ydata is None:
                self._peak_selection_start = None
                self._peak_current_profile = None
                return
        
            y_min = min(self._peak_selection_start, event.ydata)
            y_max = max(self._peak_selection_start, event.ydata)
        
            logger.info(f"Vertical profile selection: {y_min:.2f} to {y_max:.2f}")
            self._fit_vertical_peak(y_min, y_max)
    
    # Reset selection state
        self._peak_selection_start = None
        self._peak_current_profile = None

    def _fit_horizontal_peak(self, x_min, x_max):
        """Fit Gaussian to horizontal profile (TOF/Energy axis)"""
        if not self.data:
            return
    
        mode = self.mode_combo.currentIndex()
        intensity = self.data["analog"].copy() if mode == 0 else self.data["counting"].copy()
    
    # Apply sign correction
        if mode == 0:
            intensity = -intensity
        else:
            intensity = intensity
    
    # Get current axis and limits
        axis = self._compute_axis(self.data["tof"])
        ymin = int(self.spin_ymin.value())
        ymax = int(self.spin_ymax.value())
        ymin = max(0, min(ymin, intensity.shape[0]-1))
        ymax = min(ymax, intensity.shape[0])
    
    # Filter by x range
        mask = (axis >= x_min) & (axis <= x_max)
        if not mask.any():
            QMessageBox.warning(self, "No Data", "No data points in selected region")
            return
    
        x_data = axis[mask]
    
    # Calculate horizontal profile
        sliced_data = intensity[ymin:ymax, :][:, mask]
        y_data = np.mean(sliced_data, axis=0)
    
        if len(x_data) < 4:
            QMessageBox.warning(self, "Insufficient Data", "Select a wider region")
            return
    
        try:
        # Initial guess
            amp_guess = np.max(y_data) - np.min(y_data)
            center_guess = x_data[np.argmax(y_data)]
            sigma_guess = (x_max - x_min) / 4
            offset_guess = np.min(y_data)
        
            p0 = [amp_guess, center_guess, sigma_guess, offset_guess]
        
        # Fit
            popt, pcov = curve_fit(gaussian_profile, x_data, y_data, p0=p0, maxfev=5000)
            amp, center, sigma, offset = popt
            perr = np.sqrt(np.diag(pcov))
        
        # Calculate FWHM
            fwhm = calculate_fwhm(abs(sigma))
            fwhm_err = calculate_fwhm(perr[2])
        
        # Get axis label
            axis_mode = self._axis_mode()
            axis_label = {"TOF": "ns", "KE": "eV", "BE": "eV"}[axis_mode]
        
            logger.info(f"Horizontal profile peak fit:")
            logger.info(f"  Center: {center:.3f} ± {perr[1]:.3f} {axis_label}")
            logger.info(f"  FWHM: {fwhm:.3f} ± {fwhm_err:.3f} {axis_label}")
        
        # Plot fit
            x_fit = np.linspace(x_min, x_max, 200)
            y_fit = gaussian_profile(x_fit, *popt)
            line, = self.ax_hprof.plot(x_fit, y_fit, 'r-', linewidth=2, 
                                         label=f'FWHM={fwhm:.2f} {axis_label}')
            self._peak_fit_lines.append(line)
            self.ax_hprof.legend(fontsize=8)
        
        # Update label
            self.peak_result_label.setText(
                f"Horizontal Profile:\n"
                f"FWHM: {fwhm:.3f} ± {fwhm_err:.3f} {axis_label}\n"
                f"Center: {center:.3f} {axis_label}\n"
                f"Amplitude: {amp:.2e}"
            )
        
            self.btn_clear_peaks.setEnabled(True)
            self.canvas.draw_idle()
        
        except Exception as e:
            logger.exception(f"Horizontal peak fitting failed: {e}")
            QMessageBox.warning(self, "Fit Failed", f"Could not fit peak:\n{str(e)}")

    def _fit_vertical_peak(self, y_min, y_max):
        """Fit Gaussian to vertical profile (File Index axis)"""
        if not self.data:
            return
    
        mode = self.mode_combo.currentIndex()
        intensity = self.data["analog"].copy() if mode == 0 else self.data["counting"].copy()
    
    # Apply sign correction
        if mode == 0:
            intensity = -intensity
        else:
            intensity = intensity
    
    # Get current limits
        xmin = _safe_float(self.spin_xmin.value())
        xmax = _safe_float(self.spin_xmax.value())
        axis = self._compute_axis(self.data["tof"])
    
    # Filter by y range
        y_min_int = max(0, int(y_min))
        y_max_int = min(intensity.shape[0], int(y_max))
    
        if y_min_int >= y_max_int:
            QMessageBox.warning(self, "Invalid Range", "Invalid y range")
            return
    
    # Filter by x range
        idx_x = np.where((axis >= xmin) & (axis <= xmax))[0]
        if idx_x.size == 0:
            idx_x = np.arange(axis.size)
    
    # Calculate vertical profile
        sliced_data = intensity[y_min_int:y_max_int, :][:, idx_x]
        x_data = np.mean(sliced_data, axis=1)
        y_data = np.arange(y_min_int, y_max_int)
    
        if len(x_data) < 4:
            QMessageBox.warning(self, "Insufficient Data", "Select a wider region")
            return
    
        try:
        # Initial guess
            amp_guess = np.max(x_data) - np.min(x_data)
            center_guess = y_data[np.argmax(x_data)]
            sigma_guess = (y_max - y_min) / 4
            offset_guess = np.min(x_data)
        
            p0 = [amp_guess, center_guess, sigma_guess, offset_guess]
        
        # Fit
            popt, pcov = curve_fit(gaussian_profile, y_data, x_data, p0=p0, maxfev=5000)
            amp, center, sigma, offset = popt
            perr = np.sqrt(np.diag(pcov))
        
        # Calculate FWHM
            fwhm = calculate_fwhm(abs(sigma))
            fwhm_err = calculate_fwhm(perr[2])
        
            logger.info(f"Vertical profile peak fit:")
            logger.info(f"  Center: {center:.1f} ± {perr[1]:.1f} (file index)")
            logger.info(f"  FWHM: {fwhm:.1f} ± {fwhm_err:.1f} (file index)")
        
        # Plot fit
            y_fit = np.linspace(y_min, y_max, 200)
            x_fit = gaussian_profile(y_fit, *popt)
            line, = self.ax_vprof.plot(x_fit, y_fit, 'r-', linewidth=2,
                                         label=f'FWHM={fwhm:.1f} files')
            self._peak_fit_lines.append(line)
            self.ax_vprof.legend(fontsize=8)
        
        # Update label
            self.peak_result_label.setText(
                f"Vertical Profile:\n"
                f"FWHM: {fwhm:.1f} ± {fwhm_err:.1f} files\n"
                f"Center: {center:.1f} (file index)\n"
                f"Amplitude: {amp:.2e}"
            )
        
            self.btn_clear_peaks.setEnabled(True)
            self.canvas.draw_idle()
        
        except Exception as e:
            logger.exception(f"Vertical peak fitting failed: {e}")
            QMessageBox.warning(self, "Fit Failed", f"Could not fit peak:\n{str(e)}")


        

    def _on_mouse_move(self, event):
        """Handle mouse motion to display coordinates"""
        if event.inaxes == self.ax_main:
            # Main plot coordinates
            x_coord = event.xdata
            y_coord = event.ydata
            
            if x_coord is not None and y_coord is not None:
                # Format based on axis mode
                axis_mode = self._axis_mode()
                if axis_mode == "TOF":
                    x_label = f"TOF: {x_coord:.2f} ns"
                elif axis_mode == "KE":
                    x_label = f"KE: {x_coord:.3f} eV"
                else:  # BE
                    x_label = f"BE: {x_coord:.3f} eV"
                
                self.coord_label.setText(f"{x_label} | File: {int(y_coord)}")
        
        elif event.inaxes == self.ax_hprof:
            # Horizontal profile coordinates
            x_coord = event.xdata
            y_coord = event.ydata
            
            if x_coord is not None and y_coord is not None:
                axis_mode = self._axis_mode()
                if axis_mode == "TOF":
                    x_label = f"TOF: {x_coord:.2f} ns"
                elif axis_mode == "KE":
                    x_label = f"KE: {x_coord:.3f} eV"
                else:
                    x_label = f"BE: {x_coord:.3f} eV"
                
                self.coord_label.setText(f"{x_label} | Intensity: {y_coord:.2e}")
        
        elif event.inaxes == self.ax_vprof:
            # Vertical profile coordinates
            x_coord = event.xdata
            y_coord = event.ydata
            
            if x_coord is not None and y_coord is not None:
                self.coord_label.setText(f"Intensity: {x_coord:.2e} | File: {int(y_coord)}")
        
        else:
            # Not on any plot
            self.coord_label.setText("X: --- | Y: ---")

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

    def _export_plot_to_pdf(self):
        if not self.data: 
            QMessageBox.warning(self, "No Data", "Load data before exporting")
            return

        import matplotlib.pyplot as plt
        import numpy as np
        
        folder_name = os.path.basename(self.folder) if self.folder else "tof_plot"
        default_filename = f"{folder_name}_viewer-flipped.pdf"

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot as PDF",
            default_filename,
            "PDF Files (*.pdf)"
        )
        if not filename:
            return  # User cancelled

    # Repeat the map extraction logic from update_plot
        mode = self.mode_combo.currentIndex()
        intensity = self.data["analog"].copy() if mode == 0 else self.data["counting"].copy()
        tof = self.data["tof"]
        try:
            Sign = float(np.sign(intensity[0, np.argmax(np.abs(intensity[0, :]))]))
            if Sign == 0:
                Sign = 1.0
        except Exception:
            Sign = 1.0
        intensity *= Sign

        axis = self._compute_axis(tof)
        xmin = _safe_float(self.spin_xmin.value(), float(np.nanmin(axis)))
        xmax = _safe_float(self.spin_xmax.value(), float(np.nanmax(axis)))
        xmin, xmax = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        ymin = int(self.spin_ymin.value())
        ymax = int(self.spin_ymax.value())
        ymin = max(0, ymin)
        ymax = min(intensity.shape[0], ymax)
        if ymin >= ymax:
            QMessageBox.warning(self, "Invalid limits", "Ymin must be less than Ymax.")
            return

        idx_x = np.where((axis >= xmin) & (axis <= xmax))[0]
        if idx_x.size == 0:
            idx_x = np.arange(axis.size)
        x_full = axis[idx_x]

        sliced_data = intensity[ymin:ymax, :][:, idx_x]
        if not sliced_data.size or sliced_data.shape[0] == 0 or sliced_data.shape[1] == 0:
            QMessageBox.warning(self, "Export Error", "Selected export region is empty.")
            return

    # --- Profiles: RAW, NOT NORMALIZED! ---
        hprof = np.mean(sliced_data, axis=0)
        vprof = np.mean(sliced_data, axis=1)

    # --- Map normalization for display only ---
        denom = float(np.abs(np.max(sliced_data)))
        if denom == 0:
            denom = 1.0
        plotted = sliced_data / denom

    # --- Downsampling of map and profiles for speed/clarity ---
        if plotted.shape[1] > MAX_DISPLAY_COLS:
            step = max(1, plotted.shape[1] // MAX_DISPLAY_COLS)
            plotted = plotted[:, ::step]
            x_full = x_full[::step]
            hprof = hprof[::step]
        y_centers = np.arange(ymin, ymax)

    # --- Profile alignment for axis flip! ---
        flipped_data = plotted.T
        flipped_x = y_centers        # horizontal is file index
        flipped_y = x_full           # vertical is axis (TOF/KE/BE)
        flipped_hprof = vprof        # horizontal profile (of new x = file)
        flipped_vprof = hprof        # vertical profile (of new y = tof/energy)

    # --- NO DATA or AXIS FLIP for counting mode anymore --- 

    # ---- Figure ----
        fig = plt.figure(figsize=(10, 10))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 2, width_ratios=[8, 2], height_ratios=[2, 8],
                           wspace=0.05, hspace=0.05)
        ax_hprof = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0], sharex=ax_hprof)
        ax_vprof = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_cbar = fig.add_subplot(gs[0, 1])

    # --- Main Map ---
        cmap_name = GLOBAL_SETTINGS["plots"].get("Raw Avg", {}).get("cmap", "viridis")
        cmin = _safe_float(GLOBAL_SETTINGS["plots"].get("Raw Avg", {}).get("vmin", 0.0), 0.0)
        cmax = _safe_float(GLOBAL_SETTINGS["plots"].get("Raw Avg", {}).get("vmax", 0.4), 0.4)
        mesh = ax_main.pcolormesh(
            flipped_x, flipped_y, flipped_data, cmap=cmap_name, vmin=cmin, vmax=cmax, shading="auto")
        ax_main.set_xlabel("File Index")
        axis_mode = {"TOF": "TOF (ns)", "KE": "KE (eV)", "BE": "BE (eV)"}[self._axis_mode()]
        ax_main.set_ylabel(axis_mode)
        ax_main.set_xlim(flipped_x.min(), flipped_x.max())
        ax_main.set_ylim(flipped_y.max(), flipped_y.min())  # <--- This ensures y runs bottom (min) to top (max)!

    # --- Add ticks with values up to 8 for clarity ---
        from matplotlib.ticker import MaxNLocator
        ax_main.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax_main.yaxis.set_major_locator(MaxNLocator(nbins=8))
        nx_ticks = np.linspace(flipped_x.min(), flipped_x.max(), num=8, dtype=int)
        ny_ticks = np.linspace(flipped_y.min(), flipped_y.max(), num=8)
        ax_main.set_xticks(nx_ticks)
        ax_main.set_yticks(ny_ticks)
        ax_main.set_yticklabels([f"{v:.1f}" for v in ny_ticks])

    # --- Horizontal profile (top, uses RAW vprof) ---
        if len(flipped_hprof) == len(flipped_x):
            ax_hprof.plot(flipped_x, flipped_hprof, "k-", lw=0.5)
        ax_hprof.set_xlim(flipped_x.min(), flipped_x.max())
        ax_hprof.tick_params(labelbottom=False)

    # --- Vertical profile (right, uses RAW hprof) ---
        if len(flipped_vprof) == len(flipped_y):
            ax_vprof.plot(flipped_vprof, flipped_y, "k-", lw=0.5)
        ax_vprof.set_ylim(flipped_y.max(), flipped_y.min())
        ax_vprof.tick_params(labelleft=False)

        plt.colorbar(mesh, cax=ax_cbar)
        ax_cbar.set_ylabel("Normalized Intensity")
        ax_cbar.yaxis.set_label_position('right')
        ax_cbar.yaxis.tick_right()
        fig.suptitle("TOF Map (axes swapped) with matching profiles", fontsize=14)

    # Save to PDF
        try:
            fig.savefig(
                filename,
                format='pdf',
                bbox_inches='tight',
                dpi=300,
                metadata={
                    'Title': f'TOF Analysis - {os.path.basename(self.folder)}',
                    'Author': 'TOF Explorer 2026',
                    'Subject': 'Time-of-Flight Spectroscopy Data',
                    'Creator': 'Analysis2026.py'
                }
            )
            plt.close(fig)  # free memory
            QMessageBox.information(
                self,
                "Export Successful",
                f"Flipped viewer plot saved to:\n{filename}"
            )
            logger.info(f"Exported flipped plot to PDF: {filename}")

        except Exception as e:
            plt.close(fig)
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export plot:\n{str(e)}"
            )
            logger.exception("Failed to export plot to PDF")

    def _load_baseline(self):
        """Load baseline data for subtraction"""
        baseline_folder = QFileDialog.getExistingDirectory(self, "Select Baseline Folder")
        if not baseline_folder:
            return
        
        # Load baseline data
        self.progress_label.setText("Loading baseline...")
        self.pbar.setValue(10)
        
        # Create loader
        self._baseline_loader = FastLoader(baseline_folder)
        
        def on_baseline_loaded(baseline_data):
            self.pbar.setValue(50)
            if "error" in baseline_data:
                QMessageBox.critical(self, "Error Loading Baseline", baseline_data["error"])
                self.progress_label.setText("Idle")
                self.pbar.setValue(0)
                # Clean up loader
                if hasattr(self, '_baseline_loader') and self._baseline_loader is not None:
                    if self._baseline_loader.isRunning():
                        self._baseline_loader.wait()
                    self._baseline_loader.deleteLater()
                    self._baseline_loader = None
                return
            
            # Store baseline data
            self._baseline_data = baseline_data
            
            # Preserve original data if not already preserved
            if self._original_data is None:
                self._original_data = {
                    "analog": self.data["analog"].copy(),
                    "counting": self.data["counting"].copy(),
                    "tof": self.data["tof"].copy()
                }
                logger.info("Original data preserved for baseline subtraction")
            
            # Open baseline window
            try:
                self._baseline_window = BaselineWindow(self, baseline_folder, baseline_data)
                self._baseline_window.show()
            except Exception as e:
                logger.exception(f"Failed to create baseline window: {e}")
                QMessageBox.critical(self, "Error", f"Failed to open baseline window:\n{str(e)}")
            
            self.progress_label.setText("Idle")
            self.pbar.setValue(100)
            logger.info(f"Baseline loaded from: {baseline_folder}")
            
            # Clean up loader
            if hasattr(self, '_baseline_loader') and self._baseline_loader is not None:
                if self._baseline_loader.isRunning():
                    self._baseline_loader.wait()
                self._baseline_loader.deleteLater()
                self._baseline_loader = None
                
        self._baseline_loader.finished.connect(on_baseline_loaded)
        self._baseline_loader.start()
        
    
    def _apply_baseline_subtraction(self, params):
        """Apply baseline subtraction with given parameters - memory efficient"""
        if self._original_data is None:
            logger.error("Original data not available for subtraction")
            return

        baseline_data = params["baseline_data"]
        file_start = params["file_start"]
        file_end = params["file_end"]
        file_by_file = params["file_by_file"]

        # Make local copies of baseline arrays
        baseline_analog = baseline_data["analog"]
        baseline_counting = baseline_data["counting"]

        # Start from unmodified original data
        self.data["analog"] = self._original_data["analog"].copy()
        self.data["counting"] = self._original_data["counting"].copy()

        try:
            if file_by_file:
                # File-by-file subtraction - in-place, memory efficient
                max_files = min(baseline_analog.shape[0], self.data["analog"].shape[0])
                actual_end = min(file_end, max_files)
                
                logger.info(f"Applying file-by-file subtraction: files {file_start}-{actual_end}")
                
                # Process in chunks to avoid memory spikes
                chunk_size = 100
                for chunk_start in range(file_start, actual_end, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, actual_end)
                    
                    # In-place subtraction for this chunk
                    self.data["analog"][chunk_start:chunk_end, :] -= baseline_analog[chunk_start:chunk_end, :]
                    self.data["counting"][chunk_start:chunk_end, :] -= baseline_counting[chunk_start:chunk_end, :]
                    
                    # Log progress for large datasets
                    if (chunk_end - file_start) % 500 == 0:
                        logger.info(f"  Processed {chunk_end - file_start} files...")

                logger.info(f"File-by-file subtraction complete")
                
            else:
                # Total average mode - compute baseline average
                logger.info(f"Computing baseline average from files {file_start}-{file_end}")
                
                # Compute average in chunks to reduce memory usage
                n_tof = baseline_analog.shape[1]
                baseline_avg_analog = np.zeros(n_tof, dtype=np.float64)
                baseline_avg_counting = np.zeros(n_tof, dtype=np.float64)
                
                n_files = file_end - file_start
                for i in range(file_start, file_end):
                    baseline_avg_analog += baseline_analog[i, :] / n_files
                    baseline_avg_counting += baseline_counting[i, :] / n_files
                
                logger.info("Applying averaged baseline subtraction to all files")
                
                # Subtract in chunks to avoid memory spike
                chunk_size = 200
                n_main_files = self.data["analog"].shape[0]
                
                for chunk_start in range(0, n_main_files, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_main_files)
                    
                    # In-place subtraction
                    self.data["analog"][chunk_start:chunk_end, :] -= baseline_avg_analog[np.newaxis, :]
                    self.data["counting"][chunk_start:chunk_end, :] -= baseline_avg_counting[np.newaxis, :]
                    
                    # Log progress
                    if chunk_end % 1000 == 0:
                        logger.info(f"  Processed {chunk_end}/{n_main_files} files...")

                logger.info(f"Averaged baseline subtraction complete")

            # Enable reset button
            self.btn_reset_baseline.setEnabled(True)
            
            # Force garbage collection before plotting
            import gc
            gc.collect()
            
            # Update plot
            logger.info("Updating display...")
            self.update_plot()
            logger.info("Baseline subtraction successfully applied")

        except MemoryError as me:
            logger.error(f"Out of memory during subtraction: {me}")
            # Try to recover
            try:
                self.data["analog"] = self._original_data["analog"].copy()
                self.data["counting"] = self._original_data["counting"].copy()
                logger.info("Restored original data after memory error")
            except:
                pass
            
        except Exception as e:
            logger.exception(f"Baseline subtraction failed: {e}")
            # Try to recover
            try:
                self.data["analog"] = self._original_data["analog"].copy()
                self.data["counting"] = self._original_data["counting"].copy()
                logger.info("Restored original data after error")
            except:
                pass   
 
    
    def _reset_baseline(self):
        """Reset to original data (before baseline subtraction) - memory efficient"""
        if self._original_data is None:
            logger.info("No baseline to reset")
            return
        
        try:
            logger.info("Resetting to original data...")
            
            # Copy back in chunks to avoid memory spike
            chunk_size = 200
            n_files = self._original_data["analog"].shape[0]
            
            for chunk_start in range(0, n_files, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_files)
                
                # Copy chunk by chunk
                self.data["analog"][chunk_start:chunk_end, :] = self._original_data["analog"][chunk_start:chunk_end, :].copy()
                self.data["counting"][chunk_start:chunk_end, :] = self._original_data["counting"][chunk_start:chunk_end, :].copy()
                
                # Log progress for large datasets
                if chunk_end % 1000 == 0:
                    logger.info(f"  Restored {chunk_end}/{n_files} files...")
            
            logger.info("Original data restored")
            
            # Clear baseline references
            self._baseline_data = None
            self._original_data = None
            self._baseline_roi = None
            
            # Clean up loader if exists
            if hasattr(self, '_baseline_loader') and self._baseline_loader is not None:
                if self._baseline_loader.isRunning():
                    self._baseline_loader.wait()
                self._baseline_loader.deleteLater()
                self._baseline_loader = None
            
            # Disable reset button
            self.btn_reset_baseline.setEnabled(False)
            
            # Close baseline window if open
            if self._baseline_window is not None:
                self._baseline_window.close()
                self._baseline_window = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Update display
            logger.info("Updating display after reset...")
            self.update_plot()
            
            logger.info("Baseline subtraction reset complete")
            
        except MemoryError as me:
            logger.error(f"Out of memory during reset: {me}")
            # At this point data might be corrupted, warn user
            self.progress_label.setText("Memory error - restart recommended")
            
        except Exception as e:
            logger.exception(f"Reset failed: {e}")
            self.progress_label.setText("Reset failed - see console")

    
    def closeEvent(self, event):
        # Clean up baseline loader if running
        if hasattr(self, '_baseline_loader') and self._baseline_loader is not None:
            if self._baseline_loader.isRunning():
                self._baseline_loader.wait()
            self._baseline_loader.deleteLater()
        
        # Clean up main loader if running
        if hasattr(self, 'loader') and self.loader is not None:
            if self.loader.isRunning():
                self.loader.wait()
            self.loader.deleteLater()
        
        reply = QMessageBox.question(
            self,
            "Delete Configuration and Cache?",
            "Do you want to delete the saved configuration file and cache?\n\n"
            f"Config: {CONFIG_PATH}\n"
            f"Cache: processed_cache.npz files in data folders",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.No,
        )
        if reply == QMessageBox.Cancel:
            event.ignore()
            return
        elif reply == QMessageBox.Yes:
            try:
                # Delete config file
                if os.path.exists(CONFIG_PATH):
                    os.remove(CONFIG_PATH)
                    logger.info(f"Configuration file deleted: {CONFIG_PATH}")
                
                # Delete cache file from current folder
                if self.folder:
                    cache_file = os.path.join(self.folder, "processed_cache.npz")
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                        logger.info(f"Cache file deleted: {cache_file}")
                
                # Delete baseline cache if exists
                if hasattr(self, '_baseline_data') and self._baseline_data:
                    baseline_folder = self._baseline_data.get("folder")
                    if baseline_folder:
                        baseline_cache = os.path.join(baseline_folder, "processed_cache.npz")
                        if os.path.exists(baseline_cache):
                            os.remove(baseline_cache)
                            logger.info(f"Baseline cache deleted: {baseline_cache}")
                            
            except Exception as e:
                logger.exception(f"Error during cleanup: {e}")
                QMessageBox.warning(self, "Deletion Failed", str(e))
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = TOFExplorer()
    win.show()
    sys.exit(app.exec_())
