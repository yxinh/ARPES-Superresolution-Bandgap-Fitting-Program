import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import threading
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from scipy.optimize import curve_fit, root_scalar
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.integrate import cumulative_trapezoid
import scipy.stats as stats
from scipy.special import expit

# =============================================================================
# --- Physical Constants and Conversion Factors ---
# =============================================================================
KB_CONSTANT = 8.617333262145e-5  # Boltzmann constant in eV/K
FWHM_TO_SIGMA = 2.3548           

class Step2_GapFitting(ttk.Frame):
    def __init__(self, parent, controller=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller 
        
        # --- Internal Data Storage ---
        self.file_path = None
        self.I_raw, self.k_raw, self.e_raw = None, None, None
        self.I_raw_roi, self.I_shirley_bg, self.I_proc = None, None, None
        self.k_proc, self.e_proc = None, None 
        
        # Fitting ROI storage
        self.fit_k_vals, self.fit_e_vals = None, None
        self.I_fit_raw, self.I_fit_bg, self.I_fit_proc = None, None, None
        self.I_recon_gap, self.I_recon_gap_plus_bg = None, None
        self.I_diff = None  
        
        self._temp_I_raw_roi = None
        self._temp_I_bg_total = None
        
        self.T = 4.2  
        self.energy_res_sigma = 0.008 / FWHM_TO_SIGMA 
        self.bg_noise_val = 0.0  
        self.bg_noise_data = None
        self.alpha_est = 0.0     
        self.noise_data = None   
        
        # SC Fitting Results Storage
        self.kF_actual = None
        self.fit_k_points = []
        self.fit_results_gap = []
        self.fit_results_metal = []
        self.final_stats = {} 
        
        # Datastore for Step 3
        self.saved_results = {}
        
        self.show_mode = tk.StringVar(value="Full Raw Spectrum") 
        
        self._build_ui()

    # =============================================================================
    # --- Publication Ready Style Helper ---
    # =============================================================================
    def _set_scientific_style(self, ax):
        """Applies publication-quality styling to the given matplotlib axis."""
        ax.tick_params(direction='in', length=6, width=1.5, colors='k', top=True, right=True, labelsize=12)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # =============================================================================
    # --- UI Construction ---
    # =============================================================================
    def _build_ui(self):
        self.control_canvas = tk.Canvas(self, width=370) 
        self.control_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.control_canvas.yview)
        self.control_frame = ttk.Frame(self.control_canvas)
        
        self.control_window = self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        self.control_frame.bind("<Configure>", lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))
        self.control_canvas.bind("<Configure>", lambda e: self.control_canvas.itemconfig(self.control_window, width=e.width))
        
        self.control_canvas.bind('<Enter>', self._bound_to_mousewheel)
        self.control_canvas.bind('<Leave>', self._unbound_to_mousewheel)

        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.control_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        
        self.plot_frame = ttk.Frame(self, padding=10)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.plot_top_frame = ttk.Frame(self.plot_frame)
        self.plot_top_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        ttk.Label(self.plot_top_frame, text="Display View: ", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        
        self.cb_view = ttk.Combobox(self.plot_top_frame, textvariable=self.show_mode, state="readonly", width=40)
        self.cb_view['values'] = ["Full Raw Spectrum"]
        self.cb_view.current(0)
        self.cb_view.pack(side=tk.LEFT, padx=5)
        self.cb_view.bind("<<ComboboxSelected>>", self._on_display_mode_change)
        
        self.fig = plt.figure(figsize=(7, 5))
        self.ax = self.fig.add_subplot(111)
        self.divider = make_axes_locatable(self.ax)
        self.cax = self.divider.append_axes("right", size="5%", pad=0.05)
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self._build_constants_panel()
        self._build_step1_controls()
        self._build_step2_preprocessing() 
        self._build_step3_kf_search()
        self._build_step4_fitting()
        self._build_step5_saving()

    def _bound_to_mousewheel(self, event):
        self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.control_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.control_canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbound_to_mousewheel(self, event):
        self.control_canvas.unbind_all("<MouseWheel>")
        self.control_canvas.unbind_all("<Button-4>")
        self.control_canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if event.num == 4: self.control_canvas.yview_scroll(-1, "units")
        elif event.num == 5: self.control_canvas.yview_scroll(1, "units")
        elif event.delta != 0:
            if sys.platform == "darwin": self.control_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            else: self.control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_constants_panel(self):
        frame = tk.Frame(self.control_frame, bg='#e0e0e0', padx=2, pady=2, relief=tk.GROOVE, borderwidth=2)
        frame.pack(fill=tk.X, pady=(0, 5), padx=2)
        tk.Label(frame, text="Physical Constants & Params", bg='#e0e0e0', font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        tk.Label(frame, text=f"kB = {KB_CONSTANT} eV/K", bg='#e0e0e0').pack(anchor=tk.W)
        
        res_frame = tk.Frame(frame, bg='#e0e0e0'); res_frame.pack(fill=tk.X, pady=2)
        tk.Label(res_frame, text="Energy Res (FWHM, eV):", bg='#e0e0e0').pack(side=tk.LEFT)
        self.ent_res = ttk.Entry(res_frame, width=6); self.ent_res.insert(0, "0.008"); self.ent_res.pack(side=tk.LEFT, padx=2)

    def _build_step1_controls(self):
        frame = ttk.LabelFrame(self.control_frame, text="1. Load Low-T SC Band Data", padding=2)
        frame.pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(frame, text="Select .dat File", command=self.load_file).pack(fill=tk.X)
        self.lbl_file = ttk.Label(frame, text="No file selected", foreground="gray"); self.lbl_file.pack(fill=tk.X)
        
        temp_frame = ttk.Frame(frame); temp_frame.pack(fill=tk.X, pady=2)
        ttk.Label(temp_frame, text="Temperature T (K):").pack(side=tk.LEFT)
        self.ent_temp = ttk.Entry(temp_frame, width=6); self.ent_temp.insert(0, "14"); self.ent_temp.pack(side=tk.LEFT, padx=2)
        self.btn_auto_calc = ttk.Button(frame, text="Auto Calculate", command=self.auto_calculate)
        self.btn_auto_calc.pack(fill=tk.X, pady=5)
        self.btn_plot_raw = ttk.Button(frame, text="Load & Plot Spectrum", command=self.plot_raw_data, state=tk.DISABLED)
        self.btn_plot_raw.pack(fill=tk.X, pady=2)

    def _build_step2_preprocessing(self):
        self.frame_preproc = ttk.LabelFrame(self.control_frame, text="2. Preprocessing (Noise & Wide BG)", padding=2)
        self.frame_preproc.pack(fill=tk.X, pady=2, padx=2)
        
        bg_noise_frame = ttk.LabelFrame(self.frame_preproc, text="Constant Background Variance", padding=2)
        bg_noise_frame.pack(fill=tk.X, pady=2)
        bg_input_f = ttk.Frame(bg_noise_frame); bg_input_f.pack(fill=tk.X, pady=2)
        ttk.Label(bg_input_f, text="Bg Min Energy (eV):").pack(side=tk.LEFT)
        self.ent_bg_min_e = ttk.Entry(bg_input_f, width=6); self.ent_bg_min_e.pack(side=tk.LEFT, padx=5); self.ent_bg_min_e.insert(0, "0.02")
        
        bg_btn_f = ttk.Frame(bg_noise_frame); bg_btn_f.pack(fill=tk.X, pady=2)
        self.btn_bg_noise = ttk.Button(bg_btn_f, text="Estimate Bg Var.", command=self.estimate_bg_noise, state=tk.DISABLED)
        self.btn_bg_noise.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_insp_bg = ttk.Button(bg_btn_f, text="Inspect", command=self.inspect_bg_noise, state=tk.DISABLED)
        self.btn_insp_bg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        bg_res_f = ttk.Frame(bg_noise_frame); bg_res_f.pack(fill=tk.X, pady=2)
        ttk.Label(bg_res_f, text="Background Variance:").pack(side=tk.LEFT)
        self.var_bg_noise_disp = tk.StringVar(value="N/A")
        tk.Entry(bg_res_f, textvariable=self.var_bg_noise_disp, state='readonly', readonlybackground='#d3d3d3', width=10).pack(side=tk.LEFT, padx=2)
        
        noise_frame = ttk.LabelFrame(self.frame_preproc, text="Poisson Noise Estimation", padding=2)
        noise_frame.pack(fill=tk.X, pady=2)
        k_roi_frame = ttk.Frame(noise_frame); k_roi_frame.pack(fill=tk.X, pady=2)
        ttk.Label(k_roi_frame, text="Momentum ROI (Å⁻¹):").pack(side=tk.LEFT)
        self.ent_k_left = ttk.Entry(k_roi_frame, width=5); self.ent_k_left.pack(side=tk.LEFT, padx=1); self.ent_k_left.insert(0, "-0.2")
        ttk.Label(k_roi_frame, text="to").pack(side=tk.LEFT)
        self.ent_k_right = ttk.Entry(k_roi_frame, width=5); self.ent_k_right.pack(side=tk.LEFT, padx=1); self.ent_k_right.insert(0, "0.0")
        
        e_roi_frame = ttk.Frame(noise_frame); e_roi_frame.pack(fill=tk.X, pady=2)
        ttk.Label(e_roi_frame, text="Energy ROI (eV):").pack(side=tk.LEFT)
        self.ent_e_left = ttk.Entry(e_roi_frame, width=5); self.ent_e_left.pack(side=tk.LEFT, padx=1); self.ent_e_left.insert(0, "-0.15")
        ttk.Label(e_roi_frame, text="to").pack(side=tk.LEFT)
        self.ent_e_right = ttk.Entry(e_roi_frame, width=5); self.ent_e_right.pack(side=tk.LEFT, padx=1); self.ent_e_right.insert(0, "-0.10")
        
        sigma_frame = ttk.Frame(noise_frame); sigma_frame.pack(fill=tk.X, pady=2)
        ttk.Label(sigma_frame, text="Gaussian Sigma:").pack(side=tk.LEFT)
        self.ent_noise_sigma = ttk.Entry(sigma_frame, width=4); self.ent_noise_sigma.pack(side=tk.LEFT, padx=2); self.ent_noise_sigma.insert(0, "4.0")
        
        btn_n_frame = ttk.Frame(noise_frame); btn_n_frame.pack(fill=tk.X, pady=2)
        self.btn_noise = ttk.Button(btn_n_frame, text="Est. Alpha", command=self.estimate_poisson_level, state=tk.DISABLED)
        self.btn_noise.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_insp_noise = ttk.Button(btn_n_frame, text="Inspect", command=self.inspect_noise, state=tk.DISABLED)
        self.btn_insp_noise.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        alpha_frame = ttk.Frame(noise_frame); alpha_frame.pack(fill=tk.X)
        ttk.Label(alpha_frame, text="Poisson Alpha:").pack(side=tk.LEFT)
        self.var_alpha = tk.StringVar(value="N/A")
        tk.Entry(alpha_frame, textvariable=self.var_alpha, state='readonly', readonlybackground='#d3d3d3', width=8).pack(side=tk.LEFT, padx=2)
        
        shirley_frame = ttk.LabelFrame(self.frame_preproc, text="Shirley BG Removal (Wide ROI Crop)", padding=2)
        shirley_frame.pack(fill=tk.X, pady=2)
        s_k_frame = ttk.Frame(shirley_frame); s_k_frame.pack(fill=tk.X, pady=2)
        ttk.Label(s_k_frame, text="Crop Mom. (Å⁻¹):").pack(side=tk.LEFT)
        self.ent_s_k_left = ttk.Entry(s_k_frame, width=5); self.ent_s_k_left.pack(side=tk.LEFT, padx=1); self.ent_s_k_left.insert(0, "-0.15")
        ttk.Label(s_k_frame, text="to").pack(side=tk.LEFT)
        self.ent_s_k_right = ttk.Entry(s_k_frame, width=5); self.ent_s_k_right.pack(side=tk.LEFT, padx=1); self.ent_s_k_right.insert(0, "0.15")
        
        s_e_frame = ttk.Frame(shirley_frame); s_e_frame.pack(fill=tk.X, pady=2)
        ttk.Label(s_e_frame, text="Crop Energy (eV):").pack(side=tk.LEFT)
        self.ent_s_e_left = ttk.Entry(s_e_frame, width=5); self.ent_s_e_left.pack(side=tk.LEFT, padx=1); self.ent_s_e_left.insert(0, "-0.10")
        ttk.Label(s_e_frame, text="to").pack(side=tk.LEFT)
        self.ent_s_e_right = ttk.Entry(s_e_frame, width=5); self.ent_s_e_right.pack(side=tk.LEFT, padx=1); self.ent_s_e_right.insert(0, "0.03")

        s_param_frame = ttk.Frame(shirley_frame); s_param_frame.pack(fill=tk.X, pady=2)
        ttk.Label(s_param_frame, text="Max Iterations:").pack(side=tk.LEFT)
        self.ent_shirley_iter = ttk.Entry(s_param_frame, width=3); self.ent_shirley_iter.pack(side=tk.LEFT, padx=1); self.ent_shirley_iter.insert(0, "100")
        ttk.Label(s_param_frame, text="Tol:").pack(side=tk.LEFT, padx=(2,0))
        self.ent_shirley_tol = ttk.Entry(s_param_frame, width=5); self.ent_shirley_tol.pack(side=tk.LEFT, padx=1); self.ent_shirley_tol.insert(0, "1e-9")
        
        # [RESTORED]: Shirley Smooth k
        ttk.Label(s_param_frame, text="Smooth k (pts):").pack(side=tk.LEFT, padx=(2,0))
        self.ent_shirley_smooth = ttk.Entry(s_param_frame, width=4)
        self.ent_shirley_smooth.pack(side=tk.LEFT, padx=1)
        self.ent_shirley_smooth.insert(0, "4")
        
        insp_frame = ttk.Frame(shirley_frame); insp_frame.pack(fill=tk.X, pady=2)
        ttk.Label(insp_frame, text="Start k (Å⁻¹):").pack(side=tk.LEFT)
        self.ent_insp_k = ttk.Entry(insp_frame, width=5); self.ent_insp_k.pack(side=tk.LEFT, padx=1); self.ent_insp_k.insert(0, "0.0")
        ttk.Label(insp_frame, text="Step:").pack(side=tk.LEFT, padx=(2,0))
        self.ent_insp_step = ttk.Entry(insp_frame, width=3); self.ent_insp_step.pack(side=tk.LEFT, padx=1); self.ent_insp_step.insert(0, "1")
        
        self.btn_insp_shirley = ttk.Button(shirley_frame, text="Inspect Shirley Iterations", command=self.open_shirley_inspector, state=tk.DISABLED)
        self.btn_insp_shirley.pack(fill=tk.X, pady=2)
        self.btn_shirley = ttk.Button(shirley_frame, text="Remove Wide Background", command=self.run_shirley_bg, state=tk.DISABLED)
        self.btn_shirley.pack(fill=tk.X, pady=2)
        
        err_frame = ttk.Frame(shirley_frame); err_frame.pack(fill=tk.X, pady=2)
        ttk.Label(err_frame, text="Max Err:").pack(side=tk.LEFT)
        self.var_shirley_err = tk.StringVar(value="N/A")
        tk.Entry(err_frame, textvariable=self.var_shirley_err, state='readonly', readonlybackground='#d3d3d3', width=8).pack(side=tk.LEFT, padx=1)
        ttk.Label(err_frame, text="at k:").pack(side=tk.LEFT, padx=(1,0))
        self.var_shirley_err_k = tk.StringVar(value="N/A")
        tk.Entry(err_frame, textvariable=self.var_shirley_err_k, state='readonly', readonlybackground='#d3d3d3', width=6).pack(side=tk.LEFT, padx=1)

    def _build_step3_kf_search(self):
        self.frame_kf = ttk.LabelFrame(self.control_frame, text="3. Fermi Momentum (kF) Search", padding=2)
        self.frame_kf.pack(fill=tk.X, pady=2, padx=2)
        
        bracket_f = ttk.Frame(self.frame_kf); bracket_f.pack(fill=tk.X, pady=2)
        ttk.Label(bracket_f, text="Search Bracket:").pack(side=tk.LEFT)
        self.ent_kf_min = ttk.Entry(bracket_f, width=5); self.ent_kf_min.pack(side=tk.LEFT, padx=1); self.ent_kf_min.insert(0, "0.0")
        ttk.Label(bracket_f, text="to").pack(side=tk.LEFT)
        self.ent_kf_max = ttk.Entry(bracket_f, width=5); self.ent_kf_max.pack(side=tk.LEFT, padx=1); self.ent_kf_max.insert(0, "0.1")
        
        btn_f = ttk.Frame(self.frame_kf); btn_f.pack(fill=tk.X, pady=2)
        self.btn_search_kf = ttk.Button(btn_f, text="Find kF", command=self.search_kf, state=tk.DISABLED)
        self.btn_search_kf.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(btn_f, text="kF =").pack(side=tk.LEFT, padx=2)
        self.var_kf_result = tk.StringVar(value="N/A")
        tk.Entry(btn_f, textvariable=self.var_kf_result, state='readonly', readonlybackground='#d3d3d3', width=8).pack(side=tk.LEFT, padx=2)
        
        self.lbl_kf_status = ttk.Label(self.frame_kf, text="Waiting for Background Removal...", foreground="gray")
        self.lbl_kf_status.pack(fill=tk.X, pady=2)

    def _build_step4_fitting(self):
        self.frame_fit = ttk.LabelFrame(self.control_frame, text="4. Superconducting Gap Fitting", padding=2)
        self.frame_fit.pack(fill=tk.X, pady=2, padx=2)
        
        roi_f_k = ttk.Frame(self.frame_fit); roi_f_k.pack(fill=tk.X, pady=2)
        ttk.Label(roi_f_k, text="Fit Mom. (Å⁻¹):").pack(side=tk.LEFT)
        self.ent_fit_k_min = ttk.Entry(roi_f_k, width=5); self.ent_fit_k_min.pack(side=tk.LEFT, padx=1); self.ent_fit_k_min.insert(0, "0.0")
        ttk.Label(roi_f_k, text="to").pack(side=tk.LEFT)
        self.ent_fit_k_max = ttk.Entry(roi_f_k, width=5); self.ent_fit_k_max.pack(side=tk.LEFT, padx=1); self.ent_fit_k_max.insert(0, "0.1")
        
        roi_f_e = ttk.Frame(self.frame_fit); roi_f_e.pack(fill=tk.X, pady=2)
        ttk.Label(roi_f_e, text="Fit Energy (eV):").pack(side=tk.LEFT)
        self.ent_fit_e_min = ttk.Entry(roi_f_e, width=5); self.ent_fit_e_min.pack(side=tk.LEFT, padx=1); self.ent_fit_e_min.insert(0, "-0.02")
        ttk.Label(roi_f_e, text="to").pack(side=tk.LEFT)
        self.ent_fit_e_max = ttk.Entry(roi_f_e, width=5); self.ent_fit_e_max.pack(side=tk.LEFT, padx=1); self.ent_fit_e_max.insert(0, "0.01")

        self.f_guesses = ttk.LabelFrame(self.frame_fit, text="Initial Guesses (Gap & Metal)")
        self.f_guesses.pack(fill=tk.X, pady=2)
        
        ttk.Label(self.f_guesses, text="Delta (eV):").grid(row=0, column=0, padx=2, pady=2, sticky=tk.E)
        self.ent_guess_delta = ttk.Entry(self.f_guesses, width=6); self.ent_guess_delta.insert(0, "10e-3"); self.ent_guess_delta.grid(row=0, column=1)
        
        ttk.Label(self.f_guesses, text="Gamma (eV):").grid(row=0, column=2, padx=2, pady=2, sticky=tk.E)
        self.ent_guess_gamma = ttk.Entry(self.f_guesses, width=6); self.ent_guess_gamma.insert(0, "2e-3"); self.ent_guess_gamma.grid(row=0, column=3)
        
        ttk.Label(self.f_guesses, text="Amplitude:").grid(row=1, column=0, padx=2, pady=2, sticky=tk.E)
        self.ent_guess_scale = ttk.Entry(self.f_guesses, width=6); self.ent_guess_scale.insert(0, "1e-3"); self.ent_guess_scale.grid(row=1, column=1)
        
        self.btn_fit = ttk.Button(self.frame_fit, text="Run Gap & Metal Fitting (F-Test)", command=self.run_gap_fitting, state=tk.DISABLED)
        self.btn_fit.pack(fill=tk.X, pady=5)
        
        self.btn_inspect_fits = ttk.Button(self.frame_fit, text="Inspect Fits", command=self.open_fit_inspector, state=tk.DISABLED)
        self.btn_inspect_fits.pack(fill=tk.X, pady=2)

        ext_f = ttk.LabelFrame(self.frame_fit, text="Delta Extraction & Averaging", padding=2)
        ext_f.pack(fill=tk.X, pady=2)
        
        mult_f = ttk.Frame(ext_f); mult_f.pack(fill=tk.X, pady=1)
        ttk.Label(mult_f, text="Tolerance (N*sigma):").pack(side=tk.LEFT)
        self.ent_err_mult = ttk.Entry(mult_f, width=5)
        self.ent_err_mult.pack(side=tk.LEFT, padx=5)
        self.ent_err_mult.insert(0, "1.5")
        self.ent_err_mult.bind("<Return>", lambda e: self._on_display_mode_change())
        self.ent_err_mult.bind("<FocusOut>", lambda e: self._on_display_mode_change())
        
        res1_f = ttk.Frame(ext_f); res1_f.pack(fill=tk.X, pady=1)
        ttk.Label(res1_f, text="Delta at kF:").pack(side=tk.LEFT)
        self.var_delta_kf = tk.StringVar(value="N/A")
        tk.Entry(res1_f, textvariable=self.var_delta_kf, state='readonly', readonlybackground='#d3d3d3', width=18).pack(side=tk.RIGHT, padx=2)
        
        res2_f = ttk.Frame(ext_f); res2_f.pack(fill=tk.X, pady=1)
        ttk.Label(res2_f, text="Weighted Delta:").pack(side=tk.LEFT)
        self.var_delta_best = tk.StringVar(value="N/A")
        tk.Entry(res2_f, textvariable=self.var_delta_best, state='readonly', readonlybackground='#d3d3d3', width=18).pack(side=tk.RIGHT, padx=2)

        lim_f = ttk.LabelFrame(self.frame_fit, text="Plot Limits & Adjustments", padding=2)
        lim_f.pack(fill=tk.X, pady=2)
        
        lim_x = ttk.Frame(lim_f); lim_x.pack(fill=tk.X, pady=1)
        ttk.Label(lim_x, text="k X-axis: ").pack(side=tk.LEFT)
        self.ent_lim_k_min = ttk.Entry(lim_x, width=5); self.ent_lim_k_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(lim_x, text="to").pack(side=tk.LEFT)
        self.ent_lim_k_max = ttk.Entry(lim_x, width=5); self.ent_lim_k_max.pack(side=tk.LEFT, padx=1)
        
        lim_y_d = ttk.Frame(lim_f); lim_y_d.pack(fill=tk.X, pady=1)
        ttk.Label(lim_y_d, text="Δ Y-axis (meV):").pack(side=tk.LEFT)
        self.ent_lim_d_min = ttk.Entry(lim_y_d, width=5); self.ent_lim_d_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(lim_y_d, text="to").pack(side=tk.LEFT)
        self.ent_lim_d_max = ttk.Entry(lim_y_d, width=5); self.ent_lim_d_max.pack(side=tk.LEFT, padx=1)

        lim_y_g = ttk.Frame(lim_f); lim_y_g.pack(fill=tk.X, pady=1)
        ttk.Label(lim_y_g, text="Γ Y-axis (eV):").pack(side=tk.LEFT)
        self.ent_lim_g_min = ttk.Entry(lim_y_g, width=5); self.ent_lim_g_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(lim_y_g, text="to").pack(side=tk.LEFT)
        self.ent_lim_g_max = ttk.Entry(lim_y_g, width=5); self.ent_lim_g_max.pack(side=tk.LEFT, padx=1)
        
        lim_p = ttk.Frame(lim_f); lim_p.pack(fill=tk.X, pady=1)
        ttk.Label(lim_p, text="P-val Min:").pack(side=tk.LEFT)
        self.ent_lim_p_min = ttk.Entry(lim_p, width=6); self.ent_lim_p_min.insert(0, "1e-20"); self.ent_lim_p_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(lim_p, text="Thresh:").pack(side=tk.LEFT)
        self.ent_p_thresh = ttk.Entry(lim_p, width=5); self.ent_p_thresh.insert(0, "1e-10"); self.ent_p_thresh.pack(side=tk.LEFT, padx=1)
        
        for widget in [self.ent_lim_k_min, self.ent_lim_k_max, self.ent_lim_d_min, self.ent_lim_d_max, 
                       self.ent_lim_g_min, self.ent_lim_g_max, self.ent_lim_p_min, self.ent_p_thresh]:
            widget.bind("<Return>", lambda e: self._on_display_mode_change())
            widget.bind("<FocusOut>", lambda e: self._on_display_mode_change())

        self.lbl_fit_status = ttk.Label(self.frame_fit, text="Awaiting kF Search...", foreground="gray")
        self.lbl_fit_status.pack(fill=tk.X, pady=5)

    def _build_step5_saving(self):
        self.frame_save = ttk.LabelFrame(self.control_frame, text="5. Save Results for Step 3", padding=2)
        self.frame_save.pack(fill=tk.X, pady=2, padx=2)
        
        list_frame = ttk.Frame(self.frame_save)
        list_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        
        self.listbox_saved = tk.Listbox(list_frame, height=5)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox_saved.yview)
        self.listbox_saved.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox_saved.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        btn_f = ttk.Frame(self.frame_save)
        btn_f.pack(fill=tk.X, pady=2)
        
        self.btn_save_res = ttk.Button(btn_f, text="Save Result", command=self.save_current_result, state=tk.DISABLED)
        self.btn_save_res.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        
        self.btn_clear_res = ttk.Button(btn_f, text="Clear Selected", command=self.clear_selected_result)
        self.btn_clear_res.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        
        # Add an export button to save all fitted results to directory for Step 3
        self.btn_export = ttk.Button(self.frame_save, text="Export All Results", command=self.export_all_results)
        self.btn_export.pack(fill=tk.X, pady=2)  # Note: Adjust 'self' to the specific frame you want to place it in, e.g., your control frame
        
        self.btn_next_step = ttk.Button(self.frame_save, text="Proceed to Step 3 >>", command=self.go_to_step_3, state=tk.DISABLED)
        self.btn_next_step.pack(fill=tk.X, pady=5)

    # =============================================================================
    # --- File Loading & Preprocessing Logic ---
    # =============================================================================
    def _on_display_mode_change(self, event=None):
        self._update_plot(preserve_limits=False)

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("DAT files", "*.dat"), ("All files", "*.*")])
        if filepath:
            self.file_path = filepath
            self.lbl_file.config(text=filepath.split("/")[-1])
            self.btn_plot_raw.config(state=tk.NORMAL)
            
    def auto_calculate(self):
        self.plot_raw_data() 
        if getattr(self, 'I_raw', None) is None:
            return 
            
        self.btn_save_res.config(state=tk.DISABLED) 
        
        # Add all steps to the queue, including the final save
        self.auto_step_queue = [
            self.estimate_bg_noise,
            self.estimate_poisson_level,
            self.run_shirley_bg,
            self.search_kf,
            self.run_gap_fitting,
            self.save_current_result
        ]
        
        self.after(200, self._execute_next_auto_step)

    def _execute_next_auto_step(self):
        if not self.auto_step_queue:
            return
        func = self.auto_step_queue[0]
        # Pause queue execution if it's the save step and fitting is still running
        if func == self.save_current_result and str(self.btn_save_res['state']) != tk.NORMAL:
            self.after(500, self._execute_next_auto_step)
            return
            
        try:
            self.auto_step_queue.pop(0)()
            self.after(300, self._execute_next_auto_step)
            
        except Exception as e:
            messagebox.showerror("Auto Calculate Error", f"Process failed at {func.__name__}:\n{str(e)}")
            
    def plot_raw_data(self):
        try:
            self.T = float(self.ent_temp.get())
            raw_res = float(self.ent_res.get())
            self.energy_res_sigma = raw_res / FWHM_TO_SIGMA
            
            with open(self.file_path, 'r') as f:
                raw_lines = [line.rstrip('\n') for line in f if line.strip() != '']
            
            first_line = raw_lines[0].split('\t')
            e_vals_temp = np.array([float(x) for x in first_line if x.strip() != ''])
            k_vals_temp, intensity_temp = [], []
            for line in raw_lines[1:]:
                parts = line.split('\t')
                if len(parts) < 2: continue
                try: k_vals_temp.append(float(parts[1]))
                except ValueError: continue
                row = [float(x) if x.strip() != '' else 0.0 for x in parts[2:]]
                intensity_temp.append(row)
                
            k_vals_temp = np.array(k_vals_temp)
            I_ex_temp = np.array(intensity_temp, dtype=float).T
            
            if k_vals_temp[0] > k_vals_temp[-1]:
                k_vals_temp, I_ex_temp = k_vals_temp[::-1], I_ex_temp[:, ::-1]
            if e_vals_temp[0] > e_vals_temp[-1]:
                e_vals_temp, I_ex_temp = e_vals_temp[::-1], I_ex_temp[::-1, :]
            
            self.I_raw, self.k_raw, self.e_raw = I_ex_temp, k_vals_temp, e_vals_temp
            self.I_raw_roi, self.I_shirley_bg, self.I_proc = None, None, None
            
            self.cb_view['values'] = ["Full Raw Spectrum"]
            self.show_mode.set("Full Raw Spectrum")
            self._update_plot()
            
            self.btn_bg_noise.config(state=tk.NORMAL)
            self.btn_noise.config(state=tk.NORMAL)
            self.btn_shirley.config(state=tk.NORMAL)
            self.btn_insp_shirley.config(state=tk.NORMAL)
            
            self.btn_search_kf.config(state=tk.DISABLED)
            self.btn_fit.config(state=tk.DISABLED)
            self.btn_inspect_fits.config(state=tk.DISABLED)
            self.btn_save_res.config(state=tk.DISABLED)
            self.btn_next_step.config(state=tk.DISABLED)
            self.lbl_kf_status.config(text="Awaiting Background Removal...", foreground="gray")
            self.lbl_fit_status.config(text="Awaiting kF Search...", foreground="gray")
        except Exception as e:
            messagebox.showerror("Read Error", str(e))

    def estimate_bg_noise(self):
        try:
            min_e = float(self.ent_bg_min_e.get())
            bg_mask = self.e_raw > min_e
            if not np.any(bg_mask): return messagebox.showwarning("Warning", "No data points found above Bg Min Energy.")
            roi_bg = self.I_raw[bg_mask, :]
            self.bg_noise_val = np.var(roi_bg)
            self.var_bg_noise_disp.set(f"{self.bg_noise_val:.2e}")
            self.bg_noise_data = (self.k_raw, self.e_raw[bg_mask], roi_bg)
            self.btn_insp_bg.config(state=tk.NORMAL)
        except Exception as e: messagebox.showerror("Error", str(e))

    def inspect_bg_noise(self):
        if self.bg_noise_data is None: return
        k_vals, e_vals_bg, roi = self.bg_noise_data
        top = tk.Toplevel(self.winfo_toplevel()); top.title("Background Noise Region Inspector")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        im = ax.imshow(roi, aspect='auto', origin='lower', extent=[k_vals[0], k_vals[-1], e_vals_bg[0], e_vals_bg[-1]], cmap='inferno')
        fig.colorbar(im, ax=ax, label='Intensity (a.u.)')
        ax.set_xlabel(fr'Momentum k ($\mathrm{{\AA}}^{{-1}}$)', fontsize=12); ax.set_ylabel('Energy E (eV)', fontsize=12)
        ax.set_title("Background Estimation Region", fontsize=14)
        fig.tight_layout(); canvas.draw()

    def estimate_poisson_level(self):
        try:
            k_l, k_r = float(self.ent_k_left.get()), float(self.ent_k_right.get())
            e_l, e_r = float(self.ent_e_left.get()), float(self.ent_e_right.get())
            smooth_sigma = float(self.ent_noise_sigma.get()) 
            k_mask = (self.k_raw >= k_l) & (self.k_raw <= k_r)
            e_mask = (self.e_raw >= e_l) & (self.e_raw <= e_r)
            if not np.any(k_mask) or not np.any(e_mask): return messagebox.showwarning("Warning", "Selected ROI is empty!")
            roi = self.I_raw[np.ix_(e_mask, k_mask)]
            
            pad_w = int(np.ceil(4 * smooth_sigma))
            if pad_w > 0:
                padded_roi = np.pad(roi, pad_width=pad_w, mode='edge')
                smoothed_padded = gaussian_filter(padded_roi, sigma=smooth_sigma)
                roi_lp = smoothed_padded[pad_w:-pad_w, pad_w:-pad_w]
            else:
                roi_lp = gaussian_filter(roi, sigma=smooth_sigma)
                
            residual = roi - roi_lp
            mean_signal = np.mean(roi_lp[roi_lp > 0]) if roi_lp[roi_lp > 0].size > 0 else 1e-12
            self.alpha_est = np.sqrt(np.var(residual) / abs(mean_signal) + 1e-12)
            self.var_alpha.set(f"{self.alpha_est:.5f}") 
            self.noise_data = (roi, roi_lp, residual, k_mask, e_mask)
            self.btn_insp_noise.config(state=tk.NORMAL) 
        except Exception as e: messagebox.showerror("Noise Est. Error", str(e))

    def inspect_noise(self):
        if self.noise_data is not None: 
            k_mask, e_mask = self.noise_data[3], self.noise_data[4]
            k_roi, e_roi = self.k_raw[k_mask], self.e_raw[e_mask]
            top = tk.Toplevel(self.winfo_toplevel())
            top.title("Noise Estimation Details")
            top.geometry("1200x450")
            
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
            
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, top)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            im0 = axes[0].imshow(self.noise_data[0], aspect='auto', origin='lower', extent=[k_roi[0], k_roi[-1], e_roi[0], e_roi[-1]], cmap='inferno')
            axes[0].set_title("Original ROI"); fig.colorbar(im0, ax=axes[0])
            im1 = axes[1].imshow(self.noise_data[1], aspect='auto', origin='lower', extent=[k_roi[0], k_roi[-1], e_roi[0], e_roi[-1]], cmap='inferno')
            axes[1].set_title("Smoothed (Signal)"); fig.colorbar(im1, ax=axes[1])
            std_res = np.std(self.noise_data[2])
            im2 = axes[2].imshow(self.noise_data[2], aspect='auto', origin='lower', extent=[k_roi[0], k_roi[-1], e_roi[0], e_roi[-1]], cmap='coolwarm', vmin=-std_res*3, vmax=std_res*3)
            axes[2].set_title(f"Residual (Noise)\nalpha_est = {self.alpha_est:.4f}"); fig.colorbar(im2, ax=axes[2])
            
            fig.tight_layout(pad=2.0, w_pad=3.0)
            canvas.draw()

    def open_shirley_inspector(self):
        if self.I_raw is None: return
        try:
            start_k = float(self.ent_insp_k.get())
            plot_step, max_iter = int(self.ent_insp_step.get()), int(self.ent_shirley_iter.get())
            e_left, e_right, tol = float(self.ent_s_e_left.get()), float(self.ent_s_e_right.get()), float(self.ent_shirley_tol.get())
        except ValueError: return messagebox.showerror("Input Error", "Invalid parameters!")
        
        e_mask = (self.e_raw >= e_left) & (self.e_raw <= e_right)
        if not np.any(e_mask): return messagebox.showwarning("Warning", "Energy crop window out of bounds.")
        
        top = tk.Toplevel(self.winfo_toplevel())
        top.title("Shirley Iteration Inspector")
        top.geometry("1100x700")  # Start large
        
        current_k_idx = [np.argmin(np.abs(self.k_raw - start_k))]
        
        global_E_min, global_E_max = np.min(self.e_raw[e_mask]), np.max(self.e_raw[e_mask])
        global_I_min, global_I_max = np.min(self.I_raw[e_mask, :]), np.max(self.I_raw[e_mask, :])
        pad_I = (global_I_max - global_I_min) * 0.05
        
        ctrl_frame = ttk.Frame(top, padding=5)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(ctrl_frame, text="Jump to k (Å⁻¹):").pack(side=tk.LEFT)
        ent_goto = ttk.Entry(ctrl_frame, width=8); ent_goto.pack(side=tk.LEFT, padx=2)
        
        nav_frame = ttk.Frame(top, padding=5)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        btn_prev = ttk.Button(nav_frame, text="<< Prev k-slice", command=lambda: update_plot(step=-1))
        btn_prev.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        btn_next = ttk.Button(nav_frame, text="Next k-slice >>", command=lambda: update_plot(step=1))
        btn_next.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=5)

        plot_frame = ttk.Frame(top)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # [FIX 2]: Ensure right space reserved for outer legend
        fig.subplots_adjust(left=0.1, right=0.75, bottom=0.12, top=0.92) 
        
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        
        # [CRITICAL]: Toolbar packed first to BOTTOM to prevent vanishing
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def goto_val():
            try:
                target = float(ent_goto.get())
                current_k_idx[0] = np.argmin(np.abs(self.k_raw - target)); update_plot()
            except: pass
        ttk.Button(ctrl_frame, text="Go", command=goto_val).pack(side=tk.LEFT)

        def update_plot(step=0):
            current_k_idx[0] = max(0, min(len(self.k_raw) - 1, current_k_idx[0] + step))
            idx = current_k_idx[0]; actual_k = self.k_raw[idx]
            
            E_sorted, I_sorted = self.e_raw[e_mask], self.I_raw[e_mask, idx]
            sort_idx = np.argsort(E_sorted); E_sorted, I_sorted = E_sorted[sort_idx], I_sorted[sort_idx]
            N, I_left, I_right = len(E_sorted), I_sorted[0], I_sorted[-1]
            
            ax.clear(); B = np.linspace(I_left, I_right, N)
            ax.plot(E_sorted, I_sorted, label='Original $I(E)$', color='black', linewidth=2)
            
            for n in range(max_iter):
                B_old = np.copy(B); Y = np.maximum(I_sorted - B_old, 0)
                cum_int = np.zeros(N); cum_int[1:] = cumulative_trapezoid(Y, E_sorted)
                if cum_int[-1] == 0: break
                B = I_right + (I_left - I_right) * ((cum_int[-1] - cum_int) / cum_int[-1])
                if (n + 1) % plot_step == 0: ax.plot(E_sorted, B, label=f'Iter {n+1}', alpha=0.8)
                if np.max(np.abs(B - B_old)) < tol: break
                
            ax.plot(E_sorted, B, label='Final Shirley BG', color='red', linewidth=2.5)
            ax.plot(E_sorted, np.maximum(I_sorted - B, 1e-4), color='blue', linewidth=2, label='Subtracted Signal')
            
            self._set_scientific_style(ax)
            ax.set_xlabel('Energy (eV)', fontsize=14)
            ax.set_ylabel('Intensity (a.u.)', fontsize=14)
            ax.set_title(fr"Shirley BG Tuning | Momentum = {actual_k:.4f} $\mathrm{{\AA}}^{{-1}}$", fontsize=14)
            
            pad_I = (np.max(I_sorted) - np.min(I_sorted)) * 0.1
            ax.set_xlim(np.min(E_sorted), np.max(E_sorted))
            ax.set_ylim(np.min(I_sorted) - pad_I, np.max(I_sorted) + pad_I)
            
            # [FIX 2]: Legend perfectly mapped to the empty right side created by subplots_adjust
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10, frameon=True, edgecolor='black')
            
            canvas.draw()

        update_plot()

    def run_shirley_bg(self):
        try:
            k_l, k_r = float(self.ent_s_k_left.get()), float(self.ent_s_k_right.get())
            e_l, e_r = float(self.ent_s_e_left.get()), float(self.ent_s_e_right.get())
            max_iter, tol = int(self.ent_shirley_iter.get()), float(self.ent_shirley_tol.get())
            smooth_k_pts = float(self.ent_shirley_smooth.get())  # [RESTORED]
        except ValueError: return messagebox.showerror("Input Error", "Invalid parameters for Shirley!")
        
        k_mask = (self.k_raw >= k_l) & (self.k_raw <= k_r)
        e_mask = (self.e_raw >= e_l) & (self.e_raw <= e_r)
        if not np.any(k_mask) or not np.any(e_mask): return messagebox.showwarning("Warning", "Crop window out of bounds.")
        
        self.k_proc, self.e_proc = self.k_raw[k_mask], self.e_raw[e_mask]
        I_crop = self.I_raw[np.ix_(e_mask, k_mask)].copy()
        
        self.btn_shirley.config(state=tk.DISABLED, text="Processing...")
        threading.Thread(target=self._shirley_thread, args=(I_crop, max_iter, tol, smooth_k_pts), daemon=True).start()

    def _shirley_thread(self, I_crop, max_iter, tol, smooth_k_pts):
        try:
            I_bg_total = np.zeros_like(I_crop)
            all_converged, max_err_val, max_err_k_idx = True, 0.0, -1
            e = self.e_proc 
            for j in range(I_crop.shape[1]):
                y, bg = I_crop[:, j], np.zeros_like(I_crop[:, j])
                y_min = np.min(y); y_proc = y - y_min
                converged, last_diff = False, 0.0
                for _ in range(max_iter):
                    y_eff = np.maximum(y_proc - bg, 0)
                    integral = np.zeros_like(y_eff)
                    for i in range(len(y_eff)-2, -1, -1):
                        integral[i] = integral[i+1] + 0.5 * (y_eff[i+1] + y_eff[i]) * (e[i+1] - e[i])
                    if integral[0] == 0: break
                    new_bg = y_proc[-1] + ((y_proc[0] - y_proc[-1]) / integral[0]) * integral
                    last_diff = np.max(np.abs(new_bg - bg))
                    if last_diff < tol:
                        bg = new_bg; converged = True; break
                    bg = new_bg
                    
                if not converged:
                    all_converged = False
                    if last_diff > max_err_val: max_err_val, max_err_k_idx = last_diff, j
                I_bg_total[:, j] = bg + y_min
            
            # [RESTORED]
            if smooth_k_pts > 0:
                pad_w = int(np.ceil(4 * smooth_k_pts))
                if pad_w > 0:
                    padded_bg = np.pad(I_bg_total, pad_width=((0,0), (pad_w, pad_w)), mode='edge')
                    smoothed_bg = gaussian_filter1d(padded_bg, sigma=smooth_k_pts, axis=1)
                    I_bg_total_smoothed = smoothed_bg[:, pad_w:-pad_w]
                else:
                    I_bg_total_smoothed = gaussian_filter1d(I_bg_total, sigma=smooth_k_pts, axis=1)
            else:
                I_bg_total_smoothed = I_bg_total

            self._temp_I_raw_roi = I_crop
            self._temp_I_bg_total = I_bg_total_smoothed    
            err_k_val = self.k_proc[max_err_k_idx] if max_err_k_idx != -1 else None
            self.after(0, lambda: self._shirley_done(all_converged, max_err_val, err_k_val))
        except Exception as err:
            self.after(0, lambda: messagebox.showerror("Error", str(err)))
            self.after(0, lambda: self.btn_shirley.config(state=tk.NORMAL, text="Remove Wide Background"))

    def _shirley_done(self, all_converged, max_err, err_k):
        self.btn_shirley.config(state=tk.NORMAL, text="Remove Wide Background")
        self.I_raw_roi = self._temp_I_raw_roi
        self.I_shirley_bg = self._temp_I_bg_total
        self.I_proc = np.maximum(self.I_raw_roi - self.I_shirley_bg, 0) 
        
        if all_converged:
            self.var_shirley_err.set("Converged")
            self.var_shirley_err_k.set("-")
        else:
            self.var_shirley_err.set(f"{max_err:.2e}")
            self.var_shirley_err_k.set(f"{err_k:.4f}")
            
        self.cb_view['values'] = [
            "Full Raw Spectrum", 
            "Shirley ROI: Raw", 
            "Shirley ROI: Background", 
            "Shirley ROI: Processed"
        ]
        self.show_mode.set("Shirley ROI: Processed") 
        self._update_plot()
        
        self.btn_search_kf.config(state=tk.NORMAL)
        self.lbl_kf_status.config(text="Ready to search kF.", foreground="blue")

    # =============================================================================
    # --- SC Gap Fitting Core Logic ---
    # =============================================================================
    def search_kf(self):
        try:
            kf_min = float(self.ent_kf_min.get())
            kf_max = float(self.ent_kf_max.get())
            
            if not self.controller or not hasattr(self.controller, 'step1_module') or self.controller.step1_module.spline_func is None:
                self.lbl_kf_status.config(text="Error: Valid Spline function not found from Step 1!", foreground="red")
                self.var_kf_result.set("Error")
                self.btn_fit.config(state=tk.DISABLED)
                return
                
            spline_func = self.controller.step1_module.spline_func
            sol = root_scalar(spline_func, bracket=[kf_min, kf_max], method='bisect')
            self.kF_actual = sol.root
            self.var_kf_result.set(f"{self.kF_actual:.6f}")
            self.lbl_kf_status.config(text="kF found successfully.", foreground="green")
            
            self.btn_fit.config(state=tk.NORMAL)
            self.lbl_fit_status.config(text="Ready to run Gap Fitting.", foreground="blue")
                
        except Exception as e:
            self.lbl_kf_status.config(text=f"Search Failed: {str(e)}", foreground="red")
            self.var_kf_result.set("Error")
            self.btn_fit.config(state=tk.DISABLED)

    def calc_spectrum(self, e, delta, gamma, scale, edc_k, spline_func):
        dE_val = (e[-1] - e[0]) / max(len(e) - 1, 1)
        abs_dE = abs(dE_val)
        sigma_pixels = self.energy_res_sigma / abs_dE if abs_dE > 0 else 1.0

        pad_e_val = 4.0 * self.energy_res_sigma
        pad_n = int(np.ceil(pad_e_val / abs_dE)) if abs_dE > 0 else 0

        if pad_n > 0:
            e_left = np.array([e[0] - (pad_n - i) * dE_val for i in range(pad_n)])
            e_right = np.array([e[-1] + (i + 1) * dE_val for i in range(pad_n)])
            e_ext = np.concatenate((e_left, e, e_right))
        else:
            e_ext = e

        ksi_k = spline_func(edc_k)
        Ek = np.sqrt(ksi_k**2 + delta**2)
        u_k2 = 0.5 * (1 + ksi_k / Ek)
        v_k2 = 1.0 - u_k2
        
        A_tmp = scale * gamma * (u_k2 / ((e_ext - Ek)**2 + gamma**2) + v_k2 / ((e_ext + Ek)**2 + gamma**2))
        f_tmp = expit(-e_ext / (KB_CONSTANT * self.T))
        I_tmp = A_tmp * f_tmp 
        
        I_blur_ext = gaussian_filter1d(I_tmp, sigma=sigma_pixels)
        
        if pad_n > 0:
            I_blur = I_blur_ext[pad_n:-pad_n]
        else:
            I_blur = I_blur_ext
            
        return I_blur

    def run_gap_fitting(self):
        self.lbl_fit_status.config(text="Fitting Models in ROI... Please wait", foreground="orange")
        self.btn_fit.config(state=tk.DISABLED)
        threading.Thread(target=self._fitting_thread, args=(self.controller.step1_module.spline_func,), daemon=True).start()

    def _fitting_thread(self, spline_func):
        try:
            k_min, k_max = float(self.ent_fit_k_min.get()), float(self.ent_fit_k_max.get())
            e_min, e_max = float(self.ent_fit_e_min.get()), float(self.ent_fit_e_max.get())
            
            k_mask = (self.k_proc >= k_min) & (self.k_proc <= k_max)
            e_mask = (self.e_proc >= e_min) & (self.e_proc <= e_max)
            
            if not np.any(k_mask) or not np.any(e_mask):
                raise ValueError("Fitting ROI out of bounds or empty!")
            
            self.fit_k_vals = self.k_proc[k_mask]
            self.fit_e_vals = self.e_proc[e_mask]
            self.I_fit_raw = self.I_raw_roi[np.ix_(e_mask, k_mask)]
            self.I_fit_bg = self.I_shirley_bg[np.ix_(e_mask, k_mask)]
            self.I_fit_proc = self.I_proc[np.ix_(e_mask, k_mask)]
            
            N_k = len(self.fit_k_vals)
            
            d_str = self.ent_guess_delta.get().strip()
            g_str = self.ent_guess_gamma.get().strip()
            s_str = self.ent_guess_scale.get().strip()
            
            d_g = 10e-3 if d_str.lower() == 'auto' else float(d_str)
            g_g = 1e-3 if g_str.lower() == 'auto' else float(g_str)
            s_g = 0.1 if s_str.lower() == 'auto' else float(s_str)
            
            last_1 = [d_g, g_g, s_g]
            last_2 = [g_g, s_g] 
            
            delta_fit, gamma_fit = np.zeros(N_k), np.zeros(N_k)
            delta_err, gamma_err = np.zeros(N_k), np.zeros(N_k)
            RSS_gap, RSS_met = np.zeros(N_k), np.zeros(N_k)
            
            self.fit_k_points = []
            self.fit_results_gap = []
            self.fit_results_metal = []
            
            self.I_recon_gap = np.zeros_like(self.I_fit_proc)
            
            flag_1, flag_2 = True, True
            
            for a in range(N_k):
                edc_k = self.fit_k_vals[a]
                self.fit_k_points.append(edc_k)
                
                I_ori = self.I_fit_raw[:, a]
                I_edc = self.I_fit_proc[:, a]
                
                sigma_arr = np.sqrt(np.abs(I_ori) * self.alpha_est**2 + self.bg_noise_val + 1e-12)
                
                try:
                    popt_1, pcov_1 = curve_fit(
                        lambda e, d, g, s: self.calc_spectrum(e, d, g, s, edc_k, spline_func),
                        self.fit_e_vals, I_edc, p0=last_1,
                        bounds=([0, 0, 0], [max(abs(self.fit_e_vals)), np.inf, np.inf]),
                        maxfev=5000, ftol=1e-9, xtol=1e-9, gtol=1e-9, method='trf',
                        sigma=sigma_arr, absolute_sigma=True
                    )
                    flag_1 = False
                except RuntimeError:
                    popt_1, pcov_1 = [0, 0, 0], np.full((3, 3), np.inf)
                    flag_1 = True
                
                if not flag_1: last_1 = popt_1.copy()
                delta_fit[a], gamma_fit[a] = popt_1[0], popt_1[1]
                ci95_1 = 1.96 * np.sqrt(np.diag(pcov_1))
                delta_err[a], gamma_err[a] = ci95_1[0], ci95_1[1]
                I_fit_gap = self.calc_spectrum(self.fit_e_vals, *popt_1, edc_k, spline_func)
                RSS_gap[a] = np.sum(((I_edc - I_fit_gap) / sigma_arr)**2)
                
                self.I_recon_gap[:, a] = I_fit_gap
                
                try:
                    popt_2, pcov_2 = curve_fit(
                        lambda e, g, s: self.calc_spectrum(e, 0, g, s, edc_k, spline_func),
                        self.fit_e_vals, I_edc, p0=last_2,
                        bounds=([0, 0], [np.inf, np.inf]),
                        maxfev=5000, ftol=1e-9, xtol=1e-9, gtol=1e-9, method='trf',
                        sigma=sigma_arr, absolute_sigma=True
                    )
                    flag_2 = False
                except RuntimeError:
                    popt_2, pcov_2 = [0, 0], np.full((2, 2), np.inf)
                    flag_2 = True
                
                if not flag_2: last_2 = popt_2.copy()
                I_fit_met = self.calc_spectrum(self.fit_e_vals, 0, popt_2[0], popt_2[1], edc_k, spline_func)
                RSS_met[a] = np.sum(((I_edc - I_fit_met) / sigma_arr)**2)
                
                self.fit_results_gap.append({
                    'x': self.fit_e_vals, 'y_ori': I_ori, 'y_data': I_edc, 'y_fit': I_fit_gap, 
                    'popt': popt_1.copy(), 'orig_popt': popt_1.copy(), 'sigma': sigma_arr
                })
                self.fit_results_metal.append({'y_fit': I_fit_met})
                
            self.I_recon_gap_plus_bg = self.I_recon_gap + self.I_fit_bg
            self.I_diff = self.I_recon_gap_plus_bg - self.I_fit_raw
            
            n_points = len(self.fit_e_vals)
            f_stats = np.maximum(((RSS_met - RSS_gap) / 1) / (RSS_gap / (n_points - 3)), 0)
            p_vals = 1 - stats.f.cdf(f_stats, 1, n_points - 3)
            
            self.final_stats = {
                'delta_fit': delta_fit, 'delta_err': delta_err,
                'gamma_fit': gamma_fit, 'gamma_err': gamma_err,
                'RSS_gap': RSS_gap, 'RSS_met': RSS_met, 'p_vals': p_vals
            }
            
            self.after(0, self._fitting_done)
        except Exception as e:
            self.after(0, lambda: self.lbl_fit_status.config(text=f"Fitting Failed: {str(e)}", foreground="red"))
            self.after(0, lambda: self.btn_fit.config(state=tk.NORMAL))

    def _fitting_done(self):
        self.btn_fit.config(state=tk.NORMAL)
        self.btn_inspect_fits.config(state=tk.NORMAL)
        self.btn_save_res.config(state=tk.NORMAL)
        self.lbl_fit_status.config(text="Dual Model Fitting & F-Test Completed!", foreground="green")
        
        self.cb_view['values'] = [
            "Full Raw Spectrum",
            "Shirley ROI: Raw", 
            "Shirley ROI: Background", 
            "Shirley ROI: Processed",
            "Fit ROI: Raw Spectrum",
            "Fit ROI: Shirley Background",
            "Fit ROI: Processed (Signal)",
            "Fit ROI: Reconstructed 2D (Gap Model)",
            "Fit ROI: Reconstructed 2D + Background",
            "Fit ROI: Difference (Recon+BG - Raw)",
            "Fitted Delta (Δ)", 
            "Fitted Gamma (Γ)", 
            "F-Test: RSS Comparison",
            "F-Test: P-Value"
        ]
        self.show_mode.set("Fitted Delta (Δ)")
        self._update_plot()

    # =============================================================================
    # --- Helper: Calculate Weighted Delta ---
    # =============================================================================
    def _get_weighted_delta(self):
        if not self.final_stats: return None
        
        k_vals = np.array(self.fit_k_points)
        delta_vals = self.final_stats['delta_fit']
        err_vals = self.final_stats['delta_err']
        
        if self.kF_actual is not None:
            kF = self.kF_actual
        else:
            kF = (np.min(k_vals) + np.max(k_vals)) / 2.0
            
        mid_idx = np.argmin(np.abs(k_vals - kF))
        delta_mid = delta_vals[mid_idx]
        err_mid = err_vals[mid_idx]
        
        try: N_mult = float(self.ent_err_mult.get())
        except ValueError: N_mult = 2.0
            
        lower_bound = delta_mid - N_mult * err_mid
        upper_bound = delta_mid + N_mult * err_mid
        
        left_idx = mid_idx
        while left_idx > 0 and lower_bound <= delta_vals[left_idx-1] <= upper_bound:
            left_idx -= 1
            
        right_idx = mid_idx
        while right_idx < len(k_vals) - 1 and lower_bound <= delta_vals[right_idx+1] <= upper_bound:
            right_idx += 1
            
        sel_idx = slice(left_idx, right_idx + 1)
        sel_k = k_vals[sel_idx]
        sel_delta = delta_vals[sel_idx]
        sel_err = err_vals[sel_idx]
        
        weights = 1.0 / (sel_err**2 + 1e-12)
        delta_best = np.sum(weights * sel_delta) / np.sum(weights)
        error_best = np.sqrt(1.0 / np.sum(weights))
        
        dof = len(sel_delta) - 1
        if dof > 0:
            chi2_nu = np.sum(((sel_delta - delta_best) / sel_err)**2) / dof
        else:
            chi2_nu = 0.0
            
        return {
            'kF': kF, 'mid_idx': mid_idx, 'delta_mid': delta_mid, 'err_mid': err_mid,
            'sel_k': sel_k, 'sel_delta': sel_delta, 'sel_err': sel_err,
            'delta_best': delta_best, 'error_best': error_best, 'chi2_nu': chi2_nu
        }

    # =============================================================================
    # --- Plotting & Visualization ---
    # =============================================================================
    def _apply_axis_limits(self, ax, plot_type):
        try:
            x_min = self.ent_lim_k_min.get().strip()
            x_max = self.ent_lim_k_max.get().strip()
            if x_min and x_max: ax.set_xlim(float(x_min), float(x_max))
        except ValueError: pass

        if plot_type == 'delta':
            try:
                y_min = self.ent_lim_d_min.get().strip()
                y_max = self.ent_lim_d_max.get().strip()
                if y_min and y_max: ax.set_ylim(float(y_min), float(y_max))
            except ValueError: pass
        elif plot_type == 'gamma':
            try:
                y_min = self.ent_lim_g_min.get().strip()
                y_max = self.ent_lim_g_max.get().strip()
                if y_min and y_max: ax.set_ylim(float(y_min), float(y_max))
            except ValueError: pass

    def _update_plot(self, preserve_limits=False):
        mode = self.show_mode.get()
        self.fig.clf()
        
        if "Spectrum" in mode or "ROI" in mode:
            self.ax = self.fig.add_subplot(111)
            self.divider = make_axes_locatable(self.ax)
            self.cax = self.divider.append_axes("right", size="5%", pad=0.05)
            
            plot_I, plot_k, plot_e = None, None, None
            title_text = ""
            vmin_global, vmax_global = 0, 1
            cmap_to_use = 'inferno'

            if mode == "Full Raw Spectrum":
                if self.I_raw is None: return
                plot_I, plot_k, plot_e = self.I_raw, self.k_raw, self.e_raw
                title_text = f'Raw Band Spectrum (T = {self.T} K)'
                vmin_global = np.nanmin(self.I_raw)
                vmax_global = np.nanmax(self.I_raw)
            elif "Fit ROI" in mode:
                if self.fit_k_vals is None: return
                plot_k, plot_e = self.fit_k_vals, self.fit_e_vals
                vmin_global = np.nanmin(self.I_fit_raw)
                vmax_global = np.nanmax(self.I_fit_raw)
                
                if "Raw Spectrum" in mode: 
                    plot_I, title_text = self.I_fit_raw, f'Fit ROI: Raw Data (T = {self.T} K)'
                elif "Shirley Background" in mode: 
                    plot_I, title_text = self.I_fit_bg, f'Fit ROI: Background (T = {self.T} K)'
                elif "Processed" in mode: 
                    plot_I, title_text = self.I_fit_proc, f'Fit ROI: Processed Signal (T = {self.T} K)'
                elif "Reconstructed 2D (Gap" in mode:
                    plot_I, title_text = self.I_recon_gap, f'Reconstructed 2D Spectrum (Fit Model only)'
                elif "Reconstructed 2D +" in mode:
                    plot_I, title_text = self.I_recon_gap_plus_bg, f'Reconstructed Spectrum + Background'
                elif "Difference" in mode:
                    plot_I, title_text = self.I_diff, f'Fit ROI: Difference (Recon+BG - Raw)'
                    cmap_to_use = 'coolwarm'
                    abs_max = np.nanpercentile(np.abs(plot_I), 98) if plot_I is not None else 1.0
                    if abs_max == 0 or np.isnan(abs_max): abs_max = 1e-6
                    vmin_global, vmax_global = -abs_max, abs_max
            elif "Shirley ROI" in mode:
                if self.I_proc is None: return
                plot_k, plot_e = self.k_proc, self.e_proc
                vmin_global = np.nanmin(self.I_raw_roi)
                vmax_global = np.nanmax(self.I_raw_roi)
                
                if "Raw" in mode: 
                    plot_I, title_text = self.I_raw_roi, f'Shirley ROI: Raw (T = {self.T} K)'
                elif "Background" in mode: 
                    plot_I, title_text = self.I_shirley_bg, f'Shirley ROI: Background (T = {self.T} K)'
                else: 
                    plot_I, title_text = self.I_proc, f'Shirley ROI: Processed (T = {self.T} K)'

            if plot_I is None: return

            extent = [plot_k[0], plot_k[-1], plot_e[0], plot_e[-1]]
            im = self.ax.imshow(plot_I, aspect='auto', origin='lower', extent=extent, cmap=cmap_to_use, vmin=vmin_global, vmax=vmax_global)
            self.fig.colorbar(im, cax=self.cax)
            self.cax.set_ylabel('Intensity (a.u.)', fontsize=10)
            self.ax.set_xlabel(fr'Momentum k ($\mathrm{{\AA}}^{{-1}}$)', fontsize=12)
            self.ax.set_ylabel('Energy E (eV)', fontsize=12)
            self.ax.set_title(title_text, fontsize=14)
            self.fig.tight_layout()

        elif mode == "Fitted Delta (Δ)":
            if not self.final_stats: return
            
            ax = self.fig.add_subplot(111)
            k_vals = np.array(self.fit_k_points)
            
            w_res = self._get_weighted_delta()
            if w_res is None: return
            
            kF = w_res['kF']
            delta_mid, err_mid = w_res['delta_mid'], w_res['err_mid']
            sel_k = w_res['sel_k']
            delta_best, error_best = w_res['delta_best'], w_res['error_best']
            chi2_nu = w_res['chi2_nu']
                
            self.var_delta_kf.set(f"{delta_mid*1000:.2f} \u00B1 {err_mid*1000:.2f} meV")
            self.var_delta_best.set(f"{delta_best*1000:.2f} \u00B1 {error_best*1000:.2f} meV")

            delta_vals_meV = self.final_stats['delta_fit'] * 1000
            err_vals_meV = self.final_stats['delta_err'] * 1000
            
            ax.axvspan(sel_k[0], sel_k[-1], color='lightgray', alpha=0.5, lw=0)
            
            ax.errorbar(k_vals, delta_vals_meV, yerr=err_vals_meV, fmt='o', color='C0', mfc='C0', mec='C0', markersize=6, capsize=4, capthick=1.5, label=r"Fitted $\Delta$")
            ax.axvline(kF, color='k', linestyle='--', linewidth=1.5, label=fr"$k_F = {kF:.3f}$")
            ax.axhline(delta_best * 1000, color='C1', linestyle='-.', linewidth=2, label=r"Weighted $\Delta_{best}$")
            
            self._set_scientific_style(ax)
            ax.set_xlabel(fr'Momentum k ($\mathrm{{\AA}}^{{-1}}$)', fontsize=14)
            ax.set_ylabel(r'Fitted $\Delta$ (meV)', fontsize=14)
            
            ax2 = ax.twinx()
            ax2.plot(k_vals, err_vals_meV, 'r-', linewidth=2, label="Error")
            ax2.set_ylabel(r'Error $\Delta$ (meV)', fontsize=14, color='r')
            ax2.tick_params(direction='in', length=6, width=1.5, colors='r', right=True, labelsize=12)
            for spine in ax2.spines.values(): spine.set_linewidth(1.5)
            
            res_str = (
                f"Interval: [{sel_k[0]:.3f}, {sel_k[-1]:.3f}]\n"
                f"$\\Delta_F$ = {delta_mid*1000:.2f} $\\pm$ {err_mid*1000:.2f} meV\n"
                f"$\\Delta_{{best}}$ = {delta_best*1000:.2f} $\\pm$ {error_best*1000:.2f} meV\n"
                f"$\\chi^2_\\nu$ = {chi2_nu:.2f}"
            )
            
            handles, labels = ax.get_legend_handles_labels()
            proxy = mpatches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False)
            handles.append(proxy)
            labels.append(res_str)
            
            ax.legend(handles, labels, loc='best', fontsize=7, frameon=True, edgecolor='black', handlelength=1.2, labelspacing=0.3)
            
            ax.set_title("Fitted Superconducting Gap & Averaging", fontsize=14)
            ax.grid(True, alpha=0.2, linestyle='--')
            self._apply_axis_limits(ax, 'delta') 
            self.fig.tight_layout()

        elif mode == "Fitted Gamma (Γ)":
            ax = self.fig.add_subplot(111)
            ax.errorbar(self.fit_k_points, self.final_stats['gamma_fit'], yerr=self.final_stats['gamma_err'], fmt='s', color='green', markersize=6, capsize=4, capthick=1.5, label=r"Fitted $\Gamma$")
            
            if self.kF_actual is not None:
                ax.axvline(self.kF_actual, color='k', linestyle='--', linewidth=1.5, label=fr"$k_F = {self.kF_actual:.3f}$")
                
            self._set_scientific_style(ax)
            ax.set_xlabel(fr'Momentum k ($\mathrm{{\AA}}^{{-1}}$)', fontsize=14)
            ax.set_ylabel(r'$\Gamma$ (eV)', fontsize=14)
            ax.set_title("Fitted Scattering Rate", fontsize=14)
            
            ax.legend(loc='best', fontsize=7, frameon=True, edgecolor='black', handlelength=1.2, labelspacing=0.3)
            ax.grid(True, alpha=0.2, linestyle='--')
            self._apply_axis_limits(ax, 'gamma')
            self.fig.tight_layout()

        elif mode == "F-Test: RSS Comparison":
            ax = self.fig.add_subplot(111)
            k_pts = self.fit_k_points
            ax.plot(k_pts, self.final_stats['RSS_gap'], 'o-', color='blue', linewidth=2, markersize=6, label=r'Gap Model ($\Delta$ free)')
            ax.plot(k_pts, self.final_stats['RSS_met'], 's--', color='red', linewidth=2, markersize=6, label=r'Metal Model ($\Delta=0$)')
            
            if self.kF_actual is not None:
                ax.axvline(self.kF_actual, color='k', linestyle='--', linewidth=1.5, label=fr"$k_F = {self.kF_actual:.3f}$")
                
            self._set_scientific_style(ax)
            ax.set_xlabel(fr'Momentum k ($\mathrm{{\AA}}^{{-1}}$)', fontsize=14)
            ax.set_ylabel('Residual Sum of Squares (RSS)', fontsize=14)
            ax.set_title('Goodness of Fit Comparison', fontsize=14)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            ax.legend(loc='best', fontsize=7, frameon=True, edgecolor='black', handlelength=1.2, labelspacing=0.3)
            self.fig.tight_layout()

        elif mode == "F-Test: P-Value":
            ax = self.fig.add_subplot(111)
            k_pts = self.fit_k_points
            
            try: p_min = float(self.ent_lim_p_min.get())
            except ValueError: p_min = 1e-10
            try: thresh = float(self.ent_p_thresh.get())
            except ValueError: thresh = 0.01
            
            ax.semilogy(k_pts, self.final_stats['p_vals'], 'mD-', markersize=7, linewidth=2, label='F-test P-value')
            ax.axhline(y=thresh, color='black', linestyle='--', linewidth=2, label=f'Threshold ({thresh})')
            
            if self.kF_actual is not None:
                ax.axvline(self.kF_actual, color='k', linestyle='--', linewidth=1.5, label=fr"$k_F = {self.kF_actual:.3f}$")
                
            ax.fill_between(k_pts, p_min, thresh, color='green', alpha=0.1, label='Gap Significant')
            ax.fill_between(k_pts, thresh, 1.5, color='red', alpha=0.1, label='Gap Not Significant')
            ax.set_ylim(bottom=p_min, top=1.5)
            
            self._set_scientific_style(ax)
            ax.set_xlabel(fr'Momentum k ($\mathrm{{\AA}}^{{-1}}$)', fontsize=14)
            ax.set_ylabel('P-value (Log Scale)', fontsize=14)
            ax.set_title('Statistical Significance (F-Test)', fontsize=14)
            
            ax.legend(loc='best', fontsize=7, frameon=True, edgecolor='black', handlelength=1.2, labelspacing=0.3)
            ax.grid(True, which="both", ls="--", alpha=0.2)
            self.fig.tight_layout()

        self.canvas.draw()

    # ================= Inspection Tool =================
    def open_fit_inspector(self):
        if not self.fit_results_gap: return messagebox.showwarning("Warning", "No fits available to inspect.")
            
        top = tk.Toplevel(self.winfo_toplevel())
        top.title("SC Gap Fit Inspector (Gap Model Adjustment)")
        top.geometry("1100x800")
        current_idx = [0] 
        
        ctrl_frame = ttk.Frame(top, padding=5)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(ctrl_frame, text="Go to k (Å⁻¹):").pack(side=tk.LEFT)
        ent_goto = ttk.Entry(ctrl_frame, width=8); ent_goto.pack(side=tk.LEFT, padx=2)
        
        def goto_val():
            try:
                target = float(ent_goto.get())
                idx = np.argmin(np.abs(np.array(self.fit_k_points) - target))
                current_idx[0] = idx
                update_plot()
            except: pass
        ttk.Button(ctrl_frame, text="Go", command=goto_val).pack(side=tk.LEFT)
        
        slider_main_frame = ttk.LabelFrame(top, text="Dynamic Fit Adjustment (Gap Model)", padding=5)
        slider_main_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        res_var = tk.StringVar(value="Reduced Chi-Squared (Gap): N/A")
        ttk.Label(slider_main_frame, textvariable=res_var, font=("Arial", 10, "bold"), foreground="blue").pack(side=tk.TOP, pady=2)
        
        sliders_inner_frame = ttk.Frame(slider_main_frame)
        sliders_inner_frame.pack(side=tk.TOP, fill=tk.X)
        
        nav_frame = ttk.Frame(top, padding=5)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        btn_prev = ttk.Button(nav_frame, text="<< Prev k-slice", command=lambda: update_plot(step=-1))
        btn_prev.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        btn_next = ttk.Button(nav_frame, text="Next k-slice >>", command=lambda: update_plot(step=1))
        btn_next.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=5)
        
        plot_frame = ttk.Frame(top)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        param_vars = []
        line_fit_gap = None

        def clear_sliders():
            for w in sliders_inner_frame.winfo_children():
                w.destroy()
            param_vars.clear()

        def on_slider_change(*args):
            if not line_fit_gap: return
            idx = current_idx[0]
            data_g = self.fit_results_gap[idx]
            edc_k = self.fit_k_points[idx]
            
            popt_new = [v.get() for v in param_vars]
            
            spline_func = self.controller.step1_module.spline_func
            y_fit_new = self.calc_spectrum(data_g['x'], popt_new[0], popt_new[1], popt_new[2], edc_k, spline_func)
                             
            res_new = np.mean(((data_g['y_data'] - y_fit_new) / data_g['sigma'])**2)
            res_var.set(f"Reduced Chi-Sq (Gap): {res_new:.4e}")
            
            data_g['popt'] = popt_new
            data_g['y_fit'] = y_fit_new
            
            line_fit_gap[0].set_ydata(y_fit_new)
            canvas.draw_idle()

        def build_sliders(popt, names, orig_popt):
            clear_sliders()
            for i, (val, name, orig_val) in enumerate(zip(popt, names, orig_popt)):
                row = i
                col_base = 0
                
                ttk.Label(sliders_inner_frame, text=f"{name}:").grid(row=row, column=col_base, sticky=tk.E, padx=(5,2), pady=2)
                var = tk.DoubleVar(value=val)
                param_vars.append(var)
                
                v_min, v_max = orig_val * 0.5, orig_val * 1.5
                if v_min > v_max: v_min, v_max = v_max, v_min
                if v_min == v_max: v_min, v_max = 0.0, 0.01 
                
                s = ttk.Scale(sliders_inner_frame, from_=v_min, to=v_max, variable=var, command=on_slider_change)
                s.grid(row=row, column=col_base+1, sticky=tk.EW, padx=2)
                
                val_lbl = ttk.Label(sliders_inner_frame, text=f"{val:.4e}", width=10)
                val_lbl.grid(row=row, column=col_base+2, sticky=tk.W)
                
                def make_reset_cmd(v=var, orig=orig_val, l=val_lbl):
                    def cmd():
                        v.set(orig); l.config(text=f"{orig:.4e}"); on_slider_change()
                    return cmd
                    
                btn_reset = ttk.Button(sliders_inner_frame, text="Reset", width=5, command=make_reset_cmd())
                btn_reset.grid(row=row, column=col_base+3, padx=(0, 5))
                
                def update_lbl(event_val, l=val_lbl, v=var):
                    l.config(text=f"{v.get():.4e}")
                
                s.config(command=lambda e, l=val_lbl, v=var: (update_lbl(e, l, v), on_slider_change()))
            
            sliders_inner_frame.columnconfigure(1, weight=1)

        def update_plot(step=0):
            nonlocal line_fit_gap
            current_idx[0] = max(0, min(len(self.fit_results_gap) - 1, current_idx[0] + step))
            idx = current_idx[0]
            data_g = self.fit_results_gap[idx]
            data_m = self.fit_results_metal[idx]
            edc_k = self.fit_k_points[idx]
            
            ax.clear()
            ax.plot(data_g['x'], data_g['y_ori'], color='#AAAAAA', linestyle='--', linewidth=1.5, label='Original')
            ax.plot(data_g['x'], data_g['y_data'], 'o', markersize=5, color='#555555', markeredgecolor='none', alpha=0.6, label='Experiment')
            line_fit_gap = ax.plot(data_g['x'], data_g['y_fit'], color='darkblue', linestyle='-', linewidth=2, label=r'Gap Model ($\Delta$ free)')
            ax.plot(data_g['x'], data_m['y_fit'], color='firebrick', linestyle='-', linewidth=2, label=r'Metal Model ($\Delta=0$)')
            
            self._set_scientific_style(ax)
            ax.set_xlabel("Energy (eV)", fontsize=14)
            ax.set_ylabel("ARPES Intensity (a.u.)", fontsize=14)
            ax.set_title(fr"Dual Model Fit Comparison | k = {edc_k:.4f} $\mathrm{{\AA}}^{{-1}}$", fontsize=14)
            
            res_val = np.mean(((data_g['y_data'] - data_g['y_fit']) / data_g['sigma'])**2)
            res_var.set(f"Reduced Chi-Sq (Gap): {res_val:.4e}")
            
            ax.legend(loc="best", fontsize=7, frameon=True, edgecolor='black', handlelength=1.2, labelspacing=0.3)
            fig.tight_layout()
            canvas.draw()
            
            names = ["Delta (eV)", "Gamma (eV)", "Amplitude"]
            build_sliders(data_g['popt'], names, data_g['orig_popt'])

        update_plot() 

    # =============================================================================
    # --- Step 5: Save Results for Step 3 ---
    # =============================================================================
    def save_current_result(self):
        if not self.final_stats: return messagebox.showwarning("Warning", "No fit results to save!")
        
        filename = self.file_path.split("/")[-1] if self.file_path else "Unknown"
        T_val = self.T
        key = f"{filename} (T={T_val}K)"
        
        if key in self.saved_results:
            ans = messagebox.askyesno("Overwrite?", f"Result for {key} already exists. Overwrite?")
            if not ans: return
            
        w_res = self._get_weighted_delta()
        if w_res is None: return messagebox.showerror("Error", "Could not calculate weighted delta.")
        
        self.saved_results[key] = {
            'filename': filename,
            'Temperature': T_val,
            'k_points': self.fit_k_points,
            'final_stats': self.final_stats,
            'weighted_res': w_res,
            'kF': self.kF_actual
        }
        
        if key not in self.listbox_saved.get(0, tk.END):
            self.listbox_saved.insert(tk.END, key)
            
        messagebox.showinfo("Saved", f"Results for {key} saved successfully!")
        
        self.btn_next_step.config(state=tk.NORMAL)

    def clear_selected_result(self):
        sel_idx = self.listbox_saved.curselection()
        if not sel_idx: return messagebox.showwarning("Warning", "Please select a result to clear.")
        
        key = self.listbox_saved.get(sel_idx)
        if key in self.saved_results:
            del self.saved_results[key]
            
        self.listbox_saved.delete(sel_idx)
        
        if self.listbox_saved.size() == 0:
            self.btn_next_step.config(state=tk.DISABLED)
            
    def export_all_results(self):
        # Check if there are saved results in memory
        if not hasattr(self, 'saved_results') or not self.saved_results:
            messagebox.showwarning("Warning", "No results to export. Please fit and save results first.")
            return
            
        # Ask user for a parent directory where the 'result' folder will be created
        parent_dir = filedialog.askdirectory(title="Select Parent Directory to Create 'result' Folder")
        if not parent_dir: 
            return
            
        import os
        try:
            # Create a dedicated 'result' folder inside the selected parent directory
            export_dir = os.path.join(parent_dir, 'result')
            os.makedirs(export_dir, exist_ok=True)
            
            for key, res in self.saved_results.items():
                # 1. Extract data from the nested structure as defined in your code
                T_val = res.get('Temperature', 0)
                k_vals = np.array(res.get('k_points', []))
                kF_val = res.get('kF', self.kF_actual)
                kF_str = f"{kF_val:.4f}" if kF_val is not None else "N/A"
                
                # Inner level: final_stats dictionary
                stats = res.get('final_stats', {})
                delta_vals = np.array(stats.get('delta_fit', []))
                err_vals = np.array(stats.get('delta_err', []))
                
                gamma_vals = np.array(stats.get('gamma_fit', []))
                gamma_err_vals = np.array(stats.get('gamma_err', []))
                
                rss_gap = np.array(stats.get('RSS_gap', []))
                rss_met = np.array(stats.get('RSS_met', []))
                p_vals = np.array(stats.get('p_vals', []))
                
                n_pts = len(k_vals)
                if n_pts == 0:
                    continue # Skip if no momentum data
                
                # 2. Ensure all arrays are of the same length as k_vals
                # This ensures matrix alignment for np.column_stack
                if gamma_vals.ndim == 0 or len(gamma_vals) != n_pts: gamma_vals = np.full(n_pts, gamma_vals)
                if gamma_err_vals.ndim == 0 or len(gamma_err_vals) != n_pts: gamma_err_vals = np.full(n_pts, gamma_err_vals)
                if rss_gap.ndim == 0 or len(rss_gap) != n_pts: rss_gap = np.full(n_pts, rss_gap)
                if rss_met.ndim == 0 or len(rss_met) != n_pts: rss_met = np.full(n_pts, rss_met)
                if p_vals.ndim == 0 or len(p_vals) != n_pts: p_vals = np.full(n_pts, p_vals)
                
                # 3. Define the filename including the temperature
                filename = f"fit_results_{T_val}K.txt"
                file_path = os.path.join(export_dir, filename)
                
                # 4. Construct the data matrix
                # Step 3 expects: 0:k, 1:delta, 2:err, 3:gamma, 4:gamma_err, 5:RSS_gap, 6:RSS_met, 7:p_val
                export_data = np.column_stack((
                    k_vals, delta_vals, err_vals, gamma_vals, gamma_err_vals, rss_gap, rss_met, p_vals
                ))
                
                # 5. Write file with a 4-line header (updated column names)
                header_str = "Exported Fit Results\n" + \
                             f"Temperature: {T_val} K, kF: {kF_str}\n" + \
                             "--------------------------------------------------------------------------------\n" + \
                             "k_vals\tdelta_fit\tdelta_err\tgamma_fit\tgamma_err\tRSS_gap\tRSS_met\tp_vals"
                             
                np.savetxt(file_path, export_data, header=header_str, comments='', delimiter='\t', fmt='%.8e')
                
            messagebox.showinfo("Success", f"Successfully exported {len(self.saved_results)} files to:\n{export_dir}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
            
    def go_to_step_3(self):
        if self.controller and hasattr(self.controller, 'notebook') and hasattr(self.controller, 'step3_module'):
            self.controller.notebook.select(self.controller.step3_module)

# =============================================================================
# --- Standalone Test Execution ---
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("ARPES Tool - Step 2 Testing Environment")
    root.geometry("1350x850")
    
    class MockController:
        class MockStep1:
            @staticmethod
            def spline_func(k):
                return -1.0 * (k - 0.05)**2 + 0.01 
        step1_module = MockStep1()
        
        class MockStep3:
            pass
        step3_module = MockStep3()
        
        def __init__(self):
            self.notebook = None

    controller = MockController()
    notebook = ttk.Notebook(root)
    controller.notebook = notebook
    notebook.pack(fill=tk.BOTH, expand=True)
    
    tab_2_container = ttk.Frame(notebook)
    notebook.add(tab_2_container, text="Step 2: SC Gap Fitting & F-Test ")
    
    app_step2 = Step2_GapFitting(tab_2_container, controller=controller)
    app_step2.pack(fill=tk.BOTH, expand=True)
    
    root.mainloop()