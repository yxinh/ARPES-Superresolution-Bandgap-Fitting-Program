import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import threading
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.integrate import cumulative_trapezoid

# =============================================================================
# --- Physical Constants and Conversion Factors ---
# =============================================================================
KB_CONSTANT = 8.617333262145e-5  # Boltzmann constant in eV/K
FWHM_TO_SIGMA = 2.3548           

class Step1_BandExtraction(ttk.Frame):
    def __init__(self, parent, controller=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller 
        
        # --- Internal Data Storage ---
        self.file_path = None
        self.I_raw, self.k_raw, self.e_raw = None, None, None
        self.I_raw_roi = None       
        self.I_shirley_bg = None    
        self.I_proc = None          
        self.k_proc, self.e_proc = None, None 
        
        self._temp_I_raw_roi = None
        self._temp_I_bg_total = None
        
        self.T = 45.0  
        self.energy_res_sigma = 0.008 / FWHM_TO_SIGMA 
        self.bg_noise_val = 0.0  
        self.bg_noise_data = None
        self.alpha_est = 0.0     
        self.noise_data = None   
        
        self.fit_results_EDC = [] 
        self.fit_results_MDC = [] 
        self.extracted_df = pd.DataFrame() 
        self.extracted_points = [] 
        self.selected_points = []  
        self.spline_func = None    
        
        self.show_mode = tk.StringVar(value="Full Raw Spectrum") 
        self.picking_seed = False                  
        self.method_var = tk.StringVar(value="Hybrid") 
        
        self._build_ui()

    def get_results(self):
        return {
            "I_proc": self.I_proc,
            "k_proc": self.k_proc,
            "e_proc": self.e_proc,
            "spline_func": self.spline_func,
            "selected_points": self.selected_points,
            "bg_noise": self.bg_noise_val,
            "noise_alpha": self.alpha_est
        }

    # =============================================================================
    # --- Publication Ready Style Helper ---
    # =============================================================================
    def _set_scientific_style(self, ax):
        """Applies publication-quality styling to the given matplotlib axis."""
        ax.tick_params(direction='in', length=6, width=1.5, colors='k', top=True, right=True, labelsize=12)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # =============================================================================
    # --- UI Construction Methods ---
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
        
        self.fig, self.ax = plt.subplots(figsize=(6, 4.5)) 
        self.divider = make_axes_locatable(self.ax)
        self.cax = self.divider.append_axes("right", size="5%", pad=0.05)
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Label(self.plot_frame, text="💡 Tip: Right-Click always selects points and picks Seed.", foreground="gray", font=("Arial", 9, "italic")).pack(side=tk.BOTTOM, fill=tk.X, pady=2)
        
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click_plot)
        
        self._build_constants_panel()
        self._build_step1_controls()
        self._build_step2_preprocessing() 
        self._build_step3_controls()
        self._build_step4_controls()

    def _bound_to_mousewheel(self, event):
        self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.control_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.control_canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbound_to_mousewheel(self, event):
        self.control_canvas.unbind_all("<MouseWheel>")
        self.control_canvas.unbind_all("<Button-4>")
        self.control_canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.control_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.control_canvas.yview_scroll(1, "units")
        elif event.delta != 0:
            if sys.platform == "darwin":
                delta_dir = -1 if event.delta > 0 else 1
                self.control_canvas.yview_scroll(delta_dir, "units")
            else:
                self.control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_constants_panel(self):
        frame = tk.Frame(self.control_frame, bg='#e0e0e0', padx=2, pady=2, relief=tk.GROOVE, borderwidth=2)
        frame.pack(fill=tk.X, pady=(0, 5), padx=2)
        tk.Label(frame, text="Physical Constants & Params", bg='#e0e0e0', font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        tk.Label(frame, text=f"kB = {KB_CONSTANT} eV/K", bg='#e0e0e0').pack(anchor=tk.W)
        res_frame = tk.Frame(frame, bg='#e0e0e0'); res_frame.pack(fill=tk.X, pady=2)
        tk.Label(res_frame, text="Energy Res (FWHM, eV):", bg='#e0e0e0').pack(side=tk.LEFT)
        self.ent_res = ttk.Entry(res_frame, width=6); self.ent_res.insert(0, "0.008"); self.ent_res.pack(side=tk.LEFT, padx=2)

    def _build_step1_controls(self):
        frame = ttk.LabelFrame(self.control_frame, text="1. Load High-T Band Data", padding=2)
        frame.pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(frame, text="Select .dat File", command=self.load_file).pack(fill=tk.X)
        self.lbl_file = ttk.Label(frame, text="No file selected", foreground="gray"); self.lbl_file.pack(fill=tk.X)
        temp_frame = ttk.Frame(frame); temp_frame.pack(fill=tk.X, pady=2)
        ttk.Label(temp_frame, text="Temperature T (K):").pack(side=tk.LEFT)
        self.ent_temp = ttk.Entry(temp_frame, width=6); self.ent_temp.insert(0, "102.0"); self.ent_temp.pack(side=tk.LEFT, padx=2)
        self.btn_plot_raw = ttk.Button(frame, text="Load & Plot Spectrum", command=self.plot_raw_data, state=tk.DISABLED)
        self.btn_plot_raw.pack(fill=tk.X, pady=2)

    def _build_step2_preprocessing(self):
        self.frame_preproc = ttk.LabelFrame(self.control_frame, text="2. Preprocessing (Noise & BG)", padding=2)
        self.frame_preproc.pack(fill=tk.X, pady=2, padx=2)
        
        bg_noise_frame = ttk.LabelFrame(self.frame_preproc, text="Constant Background Variance", padding=2)
        bg_noise_frame.pack(fill=tk.X, pady=2)
        bg_input_f = ttk.Frame(bg_noise_frame); bg_input_f.pack(fill=tk.X, pady=2)
        ttk.Label(bg_input_f, text="Bg Min Energy (eV):").pack(side=tk.LEFT)
        self.ent_bg_min_e = ttk.Entry(bg_input_f, width=6); self.ent_bg_min_e.pack(side=tk.LEFT, padx=5); self.ent_bg_min_e.insert(0, "0.03")
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
        self.ent_s_e_right = ttk.Entry(s_e_frame, width=5); self.ent_s_e_right.pack(side=tk.LEFT, padx=1); self.ent_s_e_right.insert(0, "0.05")

        s_param_frame = ttk.Frame(shirley_frame); s_param_frame.pack(fill=tk.X, pady=2)
        ttk.Label(s_param_frame, text="Max Iterations:").pack(side=tk.LEFT)
        self.ent_shirley_iter = ttk.Entry(s_param_frame, width=3); self.ent_shirley_iter.pack(side=tk.LEFT, padx=1); self.ent_shirley_iter.insert(0, "100")
        ttk.Label(s_param_frame, text="Tol:").pack(side=tk.LEFT, padx=(2,0))
        self.ent_shirley_tol = ttk.Entry(s_param_frame, width=5); self.ent_shirley_tol.pack(side=tk.LEFT, padx=1); self.ent_shirley_tol.insert(0, "1e-9")
        
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
        self.btn_shirley = ttk.Button(shirley_frame, text="Crop ROI & Remove Background", command=self.run_shirley_bg, state=tk.DISABLED)
        self.btn_shirley.pack(fill=tk.X, pady=2)
        err_frame = ttk.Frame(shirley_frame); err_frame.pack(fill=tk.X, pady=2)
        ttk.Label(err_frame, text="Max Err:").pack(side=tk.LEFT)
        self.var_shirley_err = tk.StringVar(value="N/A")
        self.ent_shirley_err = tk.Entry(err_frame, textvariable=self.var_shirley_err, state='readonly', readonlybackground='#d3d3d3', width=8)
        self.ent_shirley_err.pack(side=tk.LEFT, padx=1)
        ttk.Label(err_frame, text="at k:").pack(side=tk.LEFT, padx=(1,0))
        self.var_shirley_err_k = tk.StringVar(value="N/A")
        self.ent_shirley_err_k = tk.Entry(err_frame, textvariable=self.var_shirley_err_k, state='readonly', readonlybackground='#d3d3d3', width=6)
        self.ent_shirley_err_k.pack(side=tk.LEFT, padx=1)

    def _build_step3_controls(self):
        self.frame_step3 = ttk.LabelFrame(self.control_frame, text="3. Extract Discrete Points", padding=2)
        self.frame_step3.pack(fill=tk.X, pady=2, padx=2)
        
        self.f_fit_roi = ttk.LabelFrame(self.frame_step3, text="Independent Fitting ROI", padding=2)
        self.f_fit_roi.pack(fill=tk.X, pady=2)
        f_k_f = ttk.Frame(self.f_fit_roi); f_k_f.pack(fill=tk.X, pady=1)
        ttk.Label(f_k_f, text="Fit Mom. (Å⁻¹):").pack(side=tk.LEFT)
        self.ent_fit_k_left = ttk.Entry(f_k_f, width=5); self.ent_fit_k_left.pack(side=tk.LEFT, padx=1); self.ent_fit_k_left.insert(0, "-0.08")
        ttk.Label(f_k_f, text="to").pack(side=tk.LEFT)
        self.ent_fit_k_right = ttk.Entry(f_k_f, width=5); self.ent_fit_k_right.pack(side=tk.LEFT, padx=1); self.ent_fit_k_right.insert(0, "0.10")
        f_e_f = ttk.Frame(self.f_fit_roi); f_e_f.pack(fill=tk.X, pady=1)
        ttk.Label(f_e_f, text="Fit Energy (eV):").pack(side=tk.LEFT)
        self.ent_fit_e_left = ttk.Entry(f_e_f, width=5); self.ent_fit_e_left.pack(side=tk.LEFT, padx=1); self.ent_fit_e_left.insert(0, "-0.03")
        ttk.Label(f_e_f, text="to").pack(side=tk.LEFT)
        self.ent_fit_e_right = ttk.Entry(f_e_f, width=5); self.ent_fit_e_right.pack(side=tk.LEFT, padx=1); self.ent_fit_e_right.insert(0, "0.02")

        for ent in [self.ent_fit_k_left, self.ent_fit_k_right, self.ent_fit_e_left, self.ent_fit_e_right]:
            ent.bind("<Return>", lambda e: self._on_fit_roi_change())
            ent.bind("<FocusOut>", lambda e: self._on_fit_roi_change())

        method_frame = ttk.Frame(self.frame_step3); method_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(method_frame, text="EDC", variable=self.method_var, value="EDC").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(method_frame, text="MDC", variable=self.method_var, value="MDC").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(method_frame, text="Hybrid", variable=self.method_var, value="Hybrid").pack(side=tk.LEFT, padx=2)
        
        self.f_edc_guess = ttk.LabelFrame(self.frame_step3, text="EDC Guesses (or 'auto')")
        self.f_edc_guess.pack(fill=tk.X, pady=2)
        ttk.Label(self.f_edc_guess, text="Amplitude:").grid(row=0, column=0, padx=1, pady=1, sticky=tk.E)
        self.ent_edc_scale = ttk.Entry(self.f_edc_guess, width=5); self.ent_edc_scale.insert(0, "auto"); self.ent_edc_scale.grid(row=0, column=1)
        ttk.Label(self.f_edc_guess, text="Peak E (eV):").grid(row=0, column=2, padx=1, pady=1, sticky=tk.E)
        self.ent_edc_e0 = ttk.Entry(self.f_edc_guess, width=5); self.ent_edc_e0.insert(0, "auto"); self.ent_edc_e0.grid(row=0, column=3)
        ttk.Label(self.f_edc_guess, text="Gamma (eV):").grid(row=1, column=0, padx=1, pady=1, sticky=tk.E)
        self.ent_edc_gamma = ttk.Entry(self.f_edc_guess, width=5); self.ent_edc_gamma.insert(0, "auto"); self.ent_edc_gamma.grid(row=1, column=1)
        
        self.f_mdc_guess = ttk.LabelFrame(self.frame_step3, text="MDC Guesses (or 'auto')")
        self.f_mdc_guess.pack(fill=tk.X, pady=2)
        ttk.Label(self.f_mdc_guess, text="Amp 1:").grid(row=0, column=0, padx=1, pady=1, sticky=tk.E)
        self.ent_mdc_a1 = ttk.Entry(self.f_mdc_guess, width=5); self.ent_mdc_a1.insert(0, "auto"); self.ent_mdc_a1.grid(row=0, column=1)
        ttk.Label(self.f_mdc_guess, text="Peak 1 (Å⁻¹):").grid(row=0, column=2, padx=1, pady=1, sticky=tk.E)
        self.ent_mdc_k1 = ttk.Entry(self.f_mdc_guess, width=5); self.ent_mdc_k1.insert(0, "auto"); self.ent_mdc_k1.grid(row=0, column=3)
        ttk.Label(self.f_mdc_guess, text="Gamma 1:").grid(row=1, column=0, padx=1, pady=1, sticky=tk.E)
        self.ent_mdc_gam1 = ttk.Entry(self.f_mdc_guess, width=5); self.ent_mdc_gam1.insert(0, "auto"); self.ent_mdc_gam1.grid(row=1, column=1)
        ttk.Label(self.f_mdc_guess, text="Amp 2:").grid(row=2, column=0, padx=1, pady=1, sticky=tk.E)
        self.ent_mdc_a2 = ttk.Entry(self.f_mdc_guess, width=5); self.ent_mdc_a2.insert(0, "auto"); self.ent_mdc_a2.grid(row=2, column=1)
        ttk.Label(self.f_mdc_guess, text="Peak 2 (Å⁻¹):").grid(row=2, column=2, padx=1, pady=1, sticky=tk.E)
        self.ent_mdc_k2 = ttk.Entry(self.f_mdc_guess, width=5); self.ent_mdc_k2.insert(0, "auto"); self.ent_mdc_k2.grid(row=2, column=3)
        ttk.Label(self.f_mdc_guess, text="Gamma 2:").grid(row=3, column=0, padx=1, pady=1, sticky=tk.E)
        self.ent_mdc_gam2 = ttk.Entry(self.f_mdc_guess, width=5); self.ent_mdc_gam2.insert(0, "auto"); self.ent_mdc_gam2.grid(row=3, column=1)

        self.method_var.trace_add("write", self._toggle_guess_entries)
        
        btn_e_frame = ttk.Frame(self.frame_step3); btn_e_frame.pack(fill=tk.X, pady=2)
        self.btn_extract = ttk.Button(btn_e_frame, text="Run Extraction", command=self.run_extraction, state=tk.DISABLED)
        self.btn_extract.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_clear_extract = ttk.Button(btn_e_frame, text="Clear", command=self.clear_extraction, state=tk.DISABLED)
        self.btn_clear_extract.pack(side=tk.LEFT, padx=1)
        
        self.btn_inspect_fits = ttk.Button(self.frame_step3, text="Inspect Fits", command=self.open_fit_inspector, state=tk.DISABLED)
        self.btn_inspect_fits.pack(fill=tk.X, pady=2)
        self.lbl_status = ttk.Label(self.frame_step3, text="Awaiting Background...", foreground="gray")
        self.lbl_status.pack(fill=tk.X)

    def _build_step4_controls(self):
        self.frame_step4 = ttk.LabelFrame(self.control_frame, text="4. Auto/Manual Selection & Spline", padding=2)
        self.frame_step4.pack(fill=tk.X, pady=2, padx=2)
        
        auto_frame = ttk.LabelFrame(self.frame_step4, text="Auto Selection (Pandas Trace)", padding=2)
        auto_frame.pack(fill=tk.X, pady=2)
        ttk.Label(auto_frame, text="Seed k (Å⁻¹):").grid(row=0, column=0, sticky=tk.W, pady=1)
        self.ent_seed_k = ttk.Entry(auto_frame, width=5); self.ent_seed_k.insert(0, "-0.05"); self.ent_seed_k.grid(row=0, column=1, padx=1)
        ttk.Label(auto_frame, text="Seed E (eV):").grid(row=0, column=2, sticky=tk.W, pady=1)
        self.ent_seed_e = ttk.Entry(auto_frame, width=5); self.ent_seed_e.insert(0, "0.006"); self.ent_seed_e.grid(row=0, column=3, padx=1)
        self.btn_pick_seed = ttk.Button(auto_frame, text="Pick Plot", command=self.toggle_pick_seed, state=tk.DISABLED, width=6)
        self.btn_pick_seed.grid(row=0, column=4, padx=2, sticky=tk.EW)
        ttk.Label(auto_frame, text="k Tol. (Å⁻¹):").grid(row=1, column=0, sticky=tk.W, pady=1)
        self.ent_k_tol = ttk.Entry(auto_frame, width=5); self.ent_k_tol.insert(0, "0.01"); self.ent_k_tol.grid(row=1, column=1, padx=1)
        ttk.Label(auto_frame, text="E Tol. (eV):").grid(row=1, column=2, sticky=tk.W, pady=1)
        self.ent_e_tol = ttk.Entry(auto_frame, width=5); self.ent_e_tol.insert(0, "0.005"); self.ent_e_tol.grid(row=1, column=3, padx=1)
        ttk.Label(auto_frame, text="SlopeTol:").grid(row=2, column=0, sticky=tk.W, pady=1)
        self.ent_slope_tol = ttk.Entry(auto_frame, width=5); self.ent_slope_tol.insert(0, "200.0"); self.ent_slope_tol.grid(row=2, column=1, padx=1)

        btn_container = ttk.Frame(auto_frame)
        btn_container.grid(row=3, column=0, columnspan=5, pady=5)
        self.btn_auto_sel = ttk.Button(btn_container, text="Run Auto Select", command=self._run_auto_select, state=tk.DISABLED, width=20)
        self.btn_auto_sel.pack(expand=True)

        man_frame = ttk.Frame(self.frame_step4)
        man_frame.pack(fill=tk.X, pady=2)
        self.btn_clear_sel = ttk.Button(man_frame, text="Clear Selection", command=self.clear_selection, state=tk.DISABLED)
        self.btn_clear_sel.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        spl_frame = ttk.Frame(self.frame_step4)
        spl_frame.pack(fill=tk.X, pady=2)
        ttk.Label(spl_frame, text="Spline Smooth (s):").pack(side=tk.LEFT, padx=1)
        self.ent_spline_s = ttk.Entry(spl_frame, width=6); self.ent_spline_s.insert(0, "5e-5"); self.ent_spline_s.pack(side=tk.LEFT, padx=1)
        self.btn_spline = ttk.Button(spl_frame, text="Fit Band Spline", command=self.fit_spline, state=tk.DISABLED)
        self.btn_spline.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        self.btn_next_step = ttk.Button(self.frame_step4, text="Proceed to Step 2 >>", command=self.go_to_step_2, state=tk.DISABLED)
        self.btn_next_step.pack(fill=tk.X, pady=5)

    def _on_fit_roi_change(self):
        if "Fitting ROI" in self.show_mode.get():
            try:
                float(self.ent_fit_k_left.get())
                float(self.ent_fit_k_right.get())
                float(self.ent_fit_e_left.get())
                float(self.ent_fit_e_right.get())
                self._update_plot(preserve_limits=False)
            except ValueError: pass

    def _toggle_guess_entries(self, *args):
        mode = self.method_var.get()
        edc_state = tk.NORMAL if mode in ["EDC", "Hybrid"] else tk.DISABLED
        mdc_state = tk.NORMAL if mode in ["MDC", "Hybrid"] else tk.DISABLED
        for w in self.f_edc_guess.winfo_children(): 
            if isinstance(w, ttk.Entry): w.config(state=edc_state)
        for w in self.f_mdc_guess.winfo_children(): 
            if isinstance(w, ttk.Entry): w.config(state=mdc_state)

    def _on_display_mode_change(self, event=None):
        self._update_plot(preserve_limits=False)

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("DAT files", "*.dat"), ("All files", "*.*")])
        if filepath:
            self.file_path = filepath
            self.lbl_file.config(text=filepath.split("/")[-1])
            self.btn_plot_raw.config(state=tk.NORMAL)

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
            
            self.extracted_df = pd.DataFrame()
            self.extracted_points, self.selected_points, self.spline_func = [], [], None
            self.fit_results_EDC, self.fit_results_MDC = [], []
            
            self.cb_view['values'] = ["Full Raw Spectrum"]
            self.show_mode.set("Full Raw Spectrum")
            
            self.btn_extract.config(state=tk.DISABLED)
            self.btn_clear_extract.config(state=tk.DISABLED)
            self.btn_inspect_fits.config(state=tk.DISABLED)
            self.btn_auto_sel.config(state=tk.DISABLED)
            self.btn_clear_sel.config(state=tk.DISABLED)
            self.btn_spline.config(state=tk.DISABLED)
            self.btn_pick_seed.config(state=tk.DISABLED)
            self.btn_next_step.config(state=tk.DISABLED)
            self.lbl_status.config(text="Awaiting Background Removal...", foreground="gray")
            
            self._update_plot(preserve_limits=False)
            self.after(100, self._enable_preprocessing_tools)
        except Exception as e: messagebox.showerror("Read Error", str(e))

    def _enable_preprocessing_tools(self):
        self.btn_bg_noise.config(state=tk.NORMAL)
        self.btn_noise.config(state=tk.NORMAL)
        self.btn_insp_shirley.config(state=tk.NORMAL) 
        self.btn_shirley.config(state=tk.NORMAL)

    def _update_plot(self, preserve_limits=False):
        if self.I_raw is None: return
        xlim = self.ax.get_xlim() if preserve_limits else None
        ylim = self.ax.get_ylim() if preserve_limits else None
        self.ax.clear(); self.cax.clear() 
        mode = self.show_mode.get()
        plot_I, plot_k, plot_e = None, None, None
        title_text = ""

        if mode == "Full Raw Spectrum":
            plot_I, plot_k, plot_e = self.I_raw, self.k_raw, self.e_raw
            title_text = f'Raw Band Spectrum (T = {self.T} K)'
        elif "Shirley ROI" in mode:
            plot_k, plot_e = self.k_proc, self.e_proc
            if "Raw" in mode: plot_I, title_text = self.I_raw_roi, f'Shirley ROI: Raw (T = {self.T} K)'
            elif "Background" in mode: plot_I, title_text = self.I_shirley_bg, f'Shirley ROI: Background (T = {self.T} K)'
            else: plot_I, title_text = self.I_proc, f'Shirley ROI: Processed (T = {self.T} K)'
        elif "Fitting ROI" in mode:
            try:
                f_k_l, f_k_r = float(self.ent_fit_k_left.get()), float(self.ent_fit_k_right.get())
                f_e_l, f_e_r = float(self.ent_fit_e_left.get()), float(self.ent_fit_e_right.get())
            except ValueError:
                self.show_mode.set("Shirley ROI: Processed")
                return self._update_plot(preserve_limits=False)

            fit_k_mask = (self.k_proc >= f_k_l) & (self.k_proc <= f_k_r)
            fit_e_mask = (self.e_proc >= f_e_l) & (self.e_proc <= f_e_r)
            if not np.any(fit_k_mask) or not np.any(fit_e_mask):
                self.show_mode.set("Shirley ROI: Processed")
                return self._update_plot(preserve_limits=False)

            plot_k, plot_e = self.k_proc[fit_k_mask], self.e_proc[fit_e_mask]
            
            if "Raw" in mode: 
                plot_I, title_text = self.I_raw_roi[np.ix_(fit_e_mask, fit_k_mask)], f'Fitting ROI: Raw (T = {self.T} K)'
            elif "Background" in mode: 
                plot_I, title_text = self.I_shirley_bg[np.ix_(fit_e_mask, fit_k_mask)], f'Fitting ROI: Background (T = {self.T} K)'
            elif "Processed" in mode: 
                plot_I, title_text = self.I_proc[np.ix_(fit_e_mask, fit_k_mask)], f'Fitting ROI: Processed (T = {self.T} K)'

        if plot_I is None: return

        if "Shirley ROI" in mode:
            vmin_global, vmax_global = np.nanmin(self.I_raw_roi), np.nanmax(self.I_raw_roi)
        elif "Fitting ROI" in mode:
            f_k_l, f_k_r = float(self.ent_fit_k_left.get()), float(self.ent_fit_k_right.get())
            f_e_l, f_e_r = float(self.ent_fit_e_left.get()), float(self.ent_fit_e_right.get())
            fit_k_mask = (self.k_proc >= f_k_l) & (self.k_proc <= f_k_r)
            fit_e_mask = (self.e_proc >= f_e_l) & (self.e_proc <= f_e_r)
            local_ref = self.I_raw_roi[np.ix_(fit_e_mask, fit_k_mask)]
            vmin_global, vmax_global = np.nanmin(local_ref), np.nanmax(local_ref)
        else:
            vmin_global, vmax_global = np.nanmin(self.I_raw), np.nanmax(self.I_raw)

        extent = [plot_k[0], plot_k[-1], plot_e[0], plot_e[-1]]
        im = self.ax.imshow(plot_I, aspect='auto', origin='lower', extent=extent, cmap='inferno', vmin=vmin_global, vmax=vmax_global)
        self.fig.colorbar(im, cax=self.cax)
        self.cax.set_ylabel('Intensity (a.u.)', fontsize=10)
        self.ax.set_xlabel(fr'Momentum k ($\mathrm{{\AA}}^{{-1}}$)', fontsize=12)
        self.ax.set_ylabel('Energy E (eV)', fontsize=12)
        self.ax.set_title(title_text, fontsize=14)
        
        if "Background" not in mode:
            if self.extracted_points:
                ks, es = zip(*self.extracted_points)
                self.ax.scatter(ks, es, c='white', s=2, alpha=0.5, label='Extracted Points')
            if self.selected_points:
                ks_sel, es_sel = zip(*self.selected_points)
                self.ax.scatter(ks_sel, es_sel, c='red', s=12, marker='o', label='Selected Points')
            if self.spline_func is not None:
                k_smooth = np.linspace(min(ks_sel), max(ks_sel), 200)
                self.ax.plot(k_smooth, self.spline_func(k_smooth), 'g-', linewidth=2, label='Fitted Spline Band')
            if self.extracted_points or self.selected_points or self.spline_func:
                self.ax.legend(loc='upper right', fontsize=7, frameon=True, facecolor='white', edgecolor='gray', framealpha=0.8)
                
        if preserve_limits and xlim is not None and ylim is not None:
            self.ax.set_xlim(xlim); self.ax.set_ylim(ylim)
        else:
            self.ax.set_xlim(plot_k[0], plot_k[-1]); self.ax.set_ylim(plot_e[0], plot_e[-1])
                
        self.fig.tight_layout(); self.canvas.draw()

    # ================= Preprocessing =================
    def estimate_bg_noise(self):
        try:
            min_e = float(self.ent_bg_min_e.get())
            bg_mask = self.e_raw > min_e
            if not np.any(bg_mask): return messagebox.showwarning("Warning", "No data points found above the specified Bg Min Energy.")
            roi_bg = self.I_raw[bg_mask, :]
            self.bg_noise_val = np.var(roi_bg)
            self.var_bg_noise_disp.set(f"{self.bg_noise_val:.2e}")
            self.bg_noise_data = (self.k_raw, self.e_raw[bg_mask], roi_bg)
            self.btn_insp_bg.config(state=tk.NORMAL)
        except Exception as e: messagebox.showerror("Error", f"Failed to estimate bg noise: {str(e)}")

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
        
        extent = [k_vals[0], k_vals[-1], e_vals_bg[0], e_vals_bg[-1]]
        im = ax.imshow(roi, aspect='auto', origin='lower', extent=extent, cmap='inferno')
        fig.colorbar(im, ax=ax, label='Intensity (a.u.)')
        ax.set_xlabel(fr'Momentum k ($\mathrm{{\AA}}^{{-1}}$)', fontsize=12)
        ax.set_ylabel('Energy E (eV)', fontsize=12)
        ax.set_title("Background Estimation Region", fontsize=14)
        fig.tight_layout()
        canvas.draw()

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
            smooth_k_pts = float(self.ent_shirley_smooth.get())
        except ValueError: return messagebox.showerror("Input Error", "Invalid parameters for Shirley!")
        
        k_mask = (self.k_raw >= k_l) & (self.k_raw <= k_r)
        e_mask = (self.e_raw >= e_l) & (self.e_raw <= e_r)
        if not np.any(k_mask) or not np.any(e_mask): return messagebox.showwarning("Warning", "Crop window out of bounds.")
        
        self.k_proc, self.e_proc = self.k_raw[k_mask], self.e_raw[e_mask]
        I_crop = self.I_raw[np.ix_(e_mask, k_mask)].copy()
        
        self.btn_shirley.config(state=tk.DISABLED, text="Processing...")
        self.var_shirley_err.set("Calculating..."); self.var_shirley_err_k.set("...")
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
                    y_eff = np.maximum(y_proc - bg, 0); integral = np.zeros_like(y_eff)
                    for i in range(len(y_eff)-2, -1, -1):
                        integral[i] = integral[i+1] + 0.5 * (y_eff[i+1] + y_eff[i]) * (e[i+1] - e[i])
                    if integral[0] == 0: break
                    new_bg = y_proc[-1] + ((y_proc[0] - y_proc[-1]) / integral[0]) * integral
                    last_diff = np.max(np.abs(new_bg - bg))
                    if last_diff < tol: bg = new_bg; converged = True; break
                    bg = new_bg
                if not converged:
                    all_converged = False
                    if last_diff > max_err_val: max_err_val, max_err_k_idx = last_diff, j
                I_bg_total[:, j] = bg + y_min
            
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
            self.after(0, lambda: self.btn_shirley.config(state=tk.NORMAL, text="Crop ROI & Remove Background"))

    def _shirley_done(self, all_converged, max_err, err_k):
        self.btn_shirley.config(state=tk.NORMAL, text="Crop ROI & Remove Background")
        self.I_raw_roi = self._temp_I_raw_roi
        self.I_shirley_bg = self._temp_I_bg_total
        self.I_proc = np.maximum(self.I_raw_roi - self.I_shirley_bg, 0) 
        if all_converged:
            self.var_shirley_err.set("Converged"); self.var_shirley_err_k.set("-")
        else:
            self.var_shirley_err.set(f"{max_err:.2e}"); self.var_shirley_err_k.set(f"{err_k:.4f}")
        self.cb_view['values'] = [
            "Full Raw Spectrum", "Shirley ROI: Raw", "Shirley ROI: Background",
            "Shirley ROI: Processed", "Fitting ROI: Raw", "Fitting ROI: Background", "Fitting ROI: Processed"
        ]
        self.show_mode.set("Shirley ROI: Processed") 
        self.clear_extraction() 
        self._update_plot()
        self.after(100, self._enable_extraction_controls)

    def _enable_extraction_controls(self):
        self.btn_extract.config(state=tk.NORMAL)
        self.btn_pick_seed.config(state=tk.NORMAL)
        self.lbl_status.config(text="Ready to Extract points.", foreground="blue")

    # ================= Advanced Extraction (Curve_fit) & Inspection =================
    def _snap_to_nearest(self, value, grid): return grid[np.argmin(np.abs(grid - value))]

    def clear_extraction(self):
        self.extracted_df = pd.DataFrame()
        self.extracted_points, self.selected_points, self.spline_func = [], [], None
        self.fit_results_EDC, self.fit_results_MDC = [], []
        self.btn_clear_extract.config(state=tk.DISABLED); self.btn_inspect_fits.config(state=tk.DISABLED)
        self.lbl_status.config(text="Ready to Extract points.", foreground="blue")
        self._update_plot(preserve_limits=True)

    def _get_float_or_auto(self, entry_widget):
        val = entry_widget.get().strip()
        return None if val.lower() == 'auto' or val == '' else float(val)

    def run_extraction(self):
        self.lbl_status.config(text="Extracting via weighted curve_fit... Please wait")
        self.btn_extract.config(state=tk.DISABLED)
        threading.Thread(target=self._extraction_thread, daemon=True).start()

    def _extraction_thread(self):
        method = self.method_var.get()
        df_list = []
        self.fit_results_EDC, self.fit_results_MDC = [], []

        try:
            f_k_l, f_k_r = float(self.ent_fit_k_left.get()), float(self.ent_fit_k_right.get())
            f_e_l, f_e_r = float(self.ent_fit_e_left.get()), float(self.ent_fit_e_right.get())
        except ValueError:
            self.after(0, lambda: messagebox.showerror("Input Error", "Invalid Fitting ROI parameters!"))
            self.after(0, lambda: self.btn_extract.config(state=tk.NORMAL))
            self.after(0, lambda: self.lbl_status.config(text="Extraction Failed"))
            return

        fit_k_mask = (self.k_proc >= f_k_l) & (self.k_proc <= f_k_r)
        fit_e_mask = (self.e_proc >= f_e_l) & (self.e_proc <= f_e_r)
        
        if not np.any(fit_k_mask) or not np.any(fit_e_mask):
            self.after(0, lambda: messagebox.showwarning("Warning", "Fitting ROI is out of bounds or empty!"))
            self.after(0, lambda: self.btn_extract.config(state=tk.NORMAL))
            self.after(0, lambda: self.lbl_status.config(text="Extraction Failed"))
            return

        k_fit_axes, e_fit_axes = self.k_proc[fit_k_mask], self.e_proc[fit_e_mask]
        I_fit_proc = self.I_proc[np.ix_(fit_e_mask, fit_k_mask)]
        I_fit_ori = self.I_raw_roi[np.ix_(fit_e_mask, fit_k_mask)]
        
        dE = abs(e_fit_axes[-1] - e_fit_axes[0]) / max(len(e_fit_axes), 1)
        sigma_pixels = self.energy_res_sigma / dE if dE > 0 else 1.0

        def EDC_fitting(E, scale, E0, gamma):
            E_kBT = np.clip(E / (KB_CONSTANT * self.T), -100, 100) 
            A = scale * gamma / ((E - E0)**2 + gamma**2) / (1 + np.exp(E_kBT))
            pad_w = int(np.ceil(4 * sigma_pixels))
            if pad_w > 0:
                padded_A = np.pad(A, pad_width=pad_w, mode='edge')
                smoothed_A = gaussian_filter1d(padded_A, sigma=sigma_pixels)
                return smoothed_A[pad_w:-pad_w]
            else:
                return gaussian_filter1d(A, sigma=sigma_pixels)

        def double_lorentzian(x, A1, x1, gamma1, A2, x2, gamma2):
            return (A1 / ((x - x1)**2 + gamma1**2) + A2 / ((x - x2)**2 + gamma2**2))

        try:
            if method in ["EDC", "Hybrid"]:
                s = self._get_float_or_auto(self.ent_edc_scale)
                e0 = self._get_float_or_auto(self.ent_edc_e0)
                g = self._get_float_or_auto(self.ent_edc_gamma)
                flag = True; last = None
                if s is not None and e0 is not None and g is not None:
                    last = [s, e0, g]; flag = False

                for i in range(len(k_fit_axes)):
                    k, edc_data, ori_data = k_fit_axes[i], I_fit_proc[:, i], I_fit_ori[:, i] 
                    if flag: last = [np.max(edc_data)*0.01, e_fit_axes[np.argmax(edc_data)], 1e-3]
                    flag = False
                    
                    sigma_arr = np.sqrt(np.abs(ori_data) * self.alpha_est**2 + self.bg_noise_val + 1e-12)
                    
                    try:
                        popt, _ = curve_fit(EDC_fitting, e_fit_axes, edc_data, p0=last, bounds=([0, np.min(e_fit_axes), 0], [np.inf, np.max(e_fit_axes), np.inf]), sigma=sigma_arr, maxfev=1000, method='trf')
                        fit_y = EDC_fitting(e_fit_axes, *popt)
                        res = np.mean(((edc_data - fit_y) / sigma_arr)**2)
                        df_list.append({'k': k, 'E': popt[1], 'residual': res, 'type': 'EDC'}) 
                        self.fit_results_EDC.append({'k': k, 'x': e_fit_axes, 'y_ori': ori_data, 'y_data': edc_data, 'y_fit': fit_y, 'popt': popt.copy(), 'orig_popt': popt.copy(), 'sigma': sigma_arr})
                        last = popt.copy()
                    except RuntimeError: flag = True 

            if method in ["MDC", "Hybrid"]:
                a1 = self._get_float_or_auto(self.ent_mdc_a1)
                k1 = self._get_float_or_auto(self.ent_mdc_k1)
                g1 = self._get_float_or_auto(self.ent_mdc_gam1)
                a2 = self._get_float_or_auto(self.ent_mdc_a2)
                k2 = self._get_float_or_auto(self.ent_mdc_k2)
                g2 = self._get_float_or_auto(self.ent_mdc_gam2)
                flag = True; last = None
                if all(v is not None for v in [a1, k1, g1, a2, k2, g2]):
                    last = [a1, k1, g1, a2, k2, g2]; flag = False

                gamma_est = 0.001
                for i in range(len(e_fit_axes)):
                    E, mdc_data, ori_data = e_fit_axes[i], I_fit_proc[i, :], I_fit_ori[i, :]
                    k_peak = k_fit_axes[np.argmax(mdc_data)]
                    if flag: last = [np.max(mdc_data)*gamma_est**2, k_fit_axes[0]+k_fit_axes[-1]-k_peak, gamma_est, np.max(mdc_data)*gamma_est**2, k_peak, gamma_est]
                    flag = False
                    
                    sigma_arr = np.sqrt(np.abs(ori_data) * self.alpha_est**2 + self.bg_noise_val + 1e-12)
                    
                    try:
                        popt, _ = curve_fit(double_lorentzian, k_fit_axes, mdc_data, p0=last, bounds=([0, np.min(k_fit_axes), 0, 0, np.min(k_fit_axes), 0], [np.inf, np.max(k_fit_axes), np.inf, np.inf, np.max(k_fit_axes), np.inf]), sigma=sigma_arr, maxfev=1000, method='trf')
                        fit_y = double_lorentzian(k_fit_axes, *popt)
                        res = np.mean(((mdc_data - fit_y) / sigma_arr)**2)
                        
                        x1, x2 = popt[1], popt[4]
                        kL, kR = (x1, x2) if x1 < x2 else (x2, x1)
                        kL_snap, kR_snap = self._snap_to_nearest(kL, k_fit_axes), self._snap_to_nearest(kR, k_fit_axes) 
                        df_list.append({'k': kL_snap, 'E': E, 'residual': res, 'type': 'MDC_L'})
                        df_list.append({'k': kR_snap, 'E': E, 'residual': res, 'type': 'MDC_R'})
                        self.fit_results_MDC.append({'E': E, 'x': k_fit_axes, 'y_ori': ori_data, 'y_data': mdc_data, 'y_fit': fit_y, 'popt': popt.copy(), 'orig_popt': popt.copy(), 'sigma': sigma_arr})
                        last = popt.copy()
                    except RuntimeError: flag = True

            self.extracted_df = pd.DataFrame(df_list).dropna(subset=['k', 'E'])
            self.extracted_points = list(zip(self.extracted_df['k'], self.extracted_df['E'])) if not self.extracted_df.empty else []
            self.after(0, self._extraction_done)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Fitting Error", str(e)))
            self.after(0, lambda: self.lbl_status.config(text="Extraction Failed"))
            self.after(0, lambda: self.btn_extract.config(state=tk.NORMAL))

    def _extraction_done(self):
        self.lbl_status.config(text="Weighted curve fitting completed!", foreground="green")
        self.cb_view['values'] = [
            "Full Raw Spectrum", "Shirley ROI: Raw", "Shirley ROI: Background", "Shirley ROI: Processed",
            "Fitting ROI: Raw", "Fitting ROI: Background", "Fitting ROI: Processed"
        ]
        self.show_mode.set("Fitting ROI: Processed")
        self._update_plot(preserve_limits=False)
        self.after(100, self._enable_selection_tools)

    def _enable_selection_tools(self):
        self.btn_extract.config(state=tk.NORMAL); self.btn_clear_extract.config(state=tk.NORMAL)
        self.btn_inspect_fits.config(state=tk.NORMAL); self.btn_clear_sel.config(state=tk.NORMAL)
        self.btn_spline.config(state=tk.NORMAL); self.btn_auto_sel.config(state=tk.NORMAL)

    # ================= Inspection Tool =================
    def open_fit_inspector(self):
        if not self.fit_results_EDC and not self.fit_results_MDC:
            return messagebox.showwarning("Warning", "No fits available to inspect.")
            
        top = tk.Toplevel(self.winfo_toplevel())
        top.title("Curve Fit Inspector")
        top.geometry("1100x800")
        
        mode_var = tk.StringVar(value="EDC" if self.fit_results_EDC else "MDC")
        current_idx = [0] 
        
        ctrl_frame = ttk.Frame(top, padding=5)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(ctrl_frame, text="View Mode:").pack(side=tk.LEFT)
        if self.fit_results_EDC: ttk.Radiobutton(ctrl_frame, text="EDC", variable=mode_var, value="EDC", command=lambda: update_plot(reset=True)).pack(side=tk.LEFT, padx=2)
        if self.fit_results_MDC: ttk.Radiobutton(ctrl_frame, text="MDC", variable=mode_var, value="MDC", command=lambda: update_plot(reset=True)).pack(side=tk.LEFT, padx=2)
        ttk.Label(ctrl_frame, text="   Go to val:").pack(side=tk.LEFT)
        ent_goto = ttk.Entry(ctrl_frame, width=8); ent_goto.pack(side=tk.LEFT, padx=2)
        
        def goto_val():
            try:
                target = float(ent_goto.get())
                ds = self.fit_results_EDC if mode_var.get() == "EDC" else self.fit_results_MDC
                key = 'k' if mode_var.get() == "EDC" else 'E'
                vals = [item[key] for item in ds]
                current_idx[0] = np.argmin(np.abs(np.array(vals) - target)); update_plot()
            except: pass
        ttk.Button(ctrl_frame, text="Go", command=goto_val).pack(side=tk.LEFT)
        
        slider_main_frame = ttk.LabelFrame(top, text="Dynamic Fit Adjustment", padding=5)
        slider_main_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        res_var = tk.StringVar(value="Reduced Chi-Squared (Residual): N/A")
        ttk.Label(slider_main_frame, textvariable=res_var, font=("Arial", 10, "bold"), foreground="blue").pack(side=tk.TOP, pady=2)
        sliders_inner_frame = ttk.Frame(slider_main_frame); sliders_inner_frame.pack(side=tk.TOP, fill=tk.X)
        
        nav_frame = ttk.Frame(top, padding=5)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        btn_prev = ttk.Button(nav_frame, text="<< Prev", command=lambda: update_plot(step=-1)); btn_prev.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        btn_next = ttk.Button(nav_frame, text="Next >>", command=lambda: update_plot(step=1)); btn_next.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=5)
        
        plot_frame = ttk.Frame(top); plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        fig, ax = plt.subplots(); canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, plot_frame); toolbar.update(); toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        param_vars = []
        line_fit = None

        def clear_sliders():
            for w in sliders_inner_frame.winfo_children(): w.destroy()
            param_vars.clear()

        def on_slider_change(*args):
            if not line_fit: return
            mode = mode_var.get()
            ds = self.fit_results_EDC if mode == "EDC" else self.fit_results_MDC
            data = ds[current_idx[0]]
            x = data['x']
            sigma_arr = data['sigma']
            popt_new = [v.get() for v in param_vars]
            
            if mode == "EDC":
                dE = abs(x[-1] - x[0]) / max(len(x), 1)
                sigma_pixels = self.energy_res_sigma / dE if dE > 0 else 1.0
                E_kBT = np.clip(x / (KB_CONSTANT * self.T), -100, 100) 
                A = popt_new[0] * popt_new[2] / ((x - popt_new[1])**2 + popt_new[2]**2) / (1 + np.exp(E_kBT))
                pad_w = int(np.ceil(4 * sigma_pixels))
                if pad_w > 0:
                    padded_A = np.pad(A, pad_width=pad_w, mode='edge')
                    smoothed_A = gaussian_filter1d(padded_A, sigma=sigma_pixels)
                    y_fit_new = smoothed_A[pad_w:-pad_w]
                else: y_fit_new = gaussian_filter1d(A, sigma=sigma_pixels)
            else:
                y_fit_new = (popt_new[0] / ((x - popt_new[1])**2 + popt_new[2]**2) + popt_new[3] / ((x - popt_new[4])**2 + popt_new[5]**2))
                             
            res_new = np.mean(((data['y_data'] - y_fit_new) / sigma_arr)**2)
            res_var.set(f"Reduced Chi-Sq: {res_new:.4e}")
            
            data['popt'] = popt_new; data['y_fit'] = y_fit_new
            line_fit[0].set_ydata(y_fit_new)
            canvas.draw_idle()

        def build_sliders(popt, names, orig_popt):
            clear_sliders()
            for i, (val, name, orig_val) in enumerate(zip(popt, names, orig_popt)):
                row = i // 2 if len(popt) == 6 else i
                col_base = (i % 2) * 4 if len(popt) == 6 else 0
                
                ttk.Label(sliders_inner_frame, text=f"{name}:").grid(row=row, column=col_base, sticky=tk.E, padx=(5,2), pady=2)
                var = tk.DoubleVar(value=val)
                param_vars.append(var)
                
                if "Peak" in name:
                    v_min = orig_val - 0.02
                    v_max = orig_val + 0.02
                else:
                    bounds = [orig_val * 0.1, orig_val * 2.0, val * 0.1, val * 2.0]
                    v_min = min(bounds)
                    v_max = max(bounds)
                
                if v_min == v_max: 
                    v_min, v_max = val - 0.02, val + 0.02
                    
                ratio = (val - v_min) / (v_max - v_min) if v_max > v_min else 0.5
                ratio = max(0.0, min(1.0, ratio)) 
                
                dummy_var = tk.DoubleVar(value=ratio)

                s = ttk.Scale(sliders_inner_frame, from_=0.0, to=1.0, variable=dummy_var)
                s.grid(row=row, column=col_base+1, sticky=tk.EW, padx=2)
                
                val_lbl = ttk.Label(sliders_inner_frame, text=f"{val:.4e}", width=10)
                val_lbl.grid(row=row, column=col_base+2, sticky=tk.W)
                
                def make_slider_cmd(real_var=var, d_var=dummy_var, mn=v_min, mx=v_max, lbl=val_lbl):
                    def cmd(event_val):
                        r = d_var.get()
                        real_val = mn + r * (mx - mn)
                        real_var.set(real_val)
                        lbl.config(text=f"{real_val:.4e}")
                        on_slider_change()
                    return cmd
                
                s.config(command=make_slider_cmd())
                
                def make_reset_cmd(real_var=var, d_var=dummy_var, orig=orig_val, mn=v_min, mx=v_max, lbl=val_lbl):
                    def cmd():
                        real_var.set(orig)
                        r_new = (orig - mn) / (mx - mn) if mx > mn else 0.5
                        d_var.set(max(0.0, min(1.0, r_new)))
                        lbl.config(text=f"{orig:.4e}")
                        on_slider_change()
                    return cmd
                    
                btn_reset = ttk.Button(sliders_inner_frame, text="Reset", width=5, command=make_reset_cmd())
                btn_reset.grid(row=row, column=col_base+3, padx=(0, 5))
            
            sliders_inner_frame.columnconfigure(1, weight=1)
            if len(popt) == 6: sliders_inner_frame.columnconfigure(5, weight=1)

        def update_plot(step=0, reset=False):
            nonlocal line_fit
            if reset: current_idx[0] = 0
            ds = self.fit_results_EDC if mode_var.get() == "EDC" else self.fit_results_MDC
            if not ds: return
                
            current_idx[0] = max(0, min(len(ds) - 1, current_idx[0] + step))
            data = ds[current_idx[0]]
            
            ax.clear()
            ax.plot(data['x'], data['y_ori'], color='#AAAAAA', linestyle='--', linewidth=1.5, label='Original')
            ax.plot(data['x'], data['y_data'], 'o', markersize=5, color='#555555', markeredgecolor='none', alpha=0.6, label='Experiment')
            line_fit = ax.plot(data['x'], data['y_fit'], color='darkblue', linestyle='-', linewidth=2, label=f'Fitting ({mode_var.get()} Peak)')
            
            self._set_scientific_style(ax)
            if mode_var.get() == "EDC":
                ax.set_xlabel("Energy (eV)", fontsize=14)
                ax.set_title(fr"EDC Fit Inspection | k = {data['k']:.4f} $\mathrm{{\AA}}^{{-1}}$", fontsize=14)
                names = ["Amplitude", "Peak E0", "Gamma"]
            else:
                ax.set_xlabel(fr"Momentum k ($\mathrm{{\AA}}^{{-1}}$)", fontsize=14)
                ax.set_title(f"MDC Fit Inspection | E = {data['E']:.4f} eV", fontsize=14)
                names = ["Amp 1", "Peak 1", "Gamma 1", "Amp 2", "Peak 2", "Gamma 2"]
                
            ax.set_ylabel("ARPES Intensity (a.u.)", fontsize=14)
            
            res_val = np.mean(((data['y_data'] - data['y_fit']) / data['sigma'])**2)
            res_var.set(f"Reduced Chi-Sq: {res_val:.4e}")
            
            ax.legend(loc="best", fontsize=7, frameon=True, edgecolor='black', handlelength=1.2, labelspacing=0.3)
            fig.tight_layout()
            canvas.draw()
            
            build_sliders(data['popt'], names, data['orig_popt'])

        update_plot() 

    # ================= Interaction & Pandas Trace =================
    def toggle_pick_seed(self):
        self.picking_seed = not self.picking_seed
        if self.picking_seed: self.btn_pick_seed.config(text="Right-Click Plot...")
        else: self.btn_pick_seed.config(text="Pick Plot")

    def on_click_plot(self, event):
        if event.inaxes != self.ax: return
        if self.toolbar.mode != '' and event.button != 3: return
            
        if self.picking_seed and event.xdata is not None and event.ydata is not None:
            self.ent_seed_k.delete(0, tk.END); self.ent_seed_k.insert(0, f"{event.xdata:.4f}")
            self.ent_seed_e.delete(0, tk.END); self.ent_seed_e.insert(0, f"{event.ydata:.4f}")
            self.picking_seed = False; self.btn_pick_seed.config(text="Pick Plot")
            return
            
        if not self.extracted_points or event.x is None or event.y is None: return
        pts_data = np.array(self.extracted_points)
        pts_screen = self.ax.transData.transform(pts_data)
        click_screen = np.array([event.x, event.y])
        distances = np.sqrt(np.sum((pts_screen - click_screen)**2, axis=1))
        nearest_idx = np.argmin(distances)
        
        if distances[nearest_idx] < 15.0: 
            new_pt = tuple(self.extracted_points[nearest_idx])
            if new_pt in self.selected_points: self.selected_points.remove(new_pt) 
            else: self.selected_points.append(new_pt) 
            self.selected_points.sort(key=lambda x: x[0]) 
            self._update_plot(preserve_limits=True) 

    def _run_auto_select(self):
        if self.extracted_df.empty: return messagebox.showwarning("Warning", "No extracted points available!")
        try:
            seed_k_input, seed_E_input = float(self.ent_seed_k.get()), float(self.ent_seed_e.get())
            E_tol, k_tol, slope_tol = float(self.ent_e_tol.get()), float(self.ent_k_tol.get()), float(self.ent_slope_tol.get())
        except ValueError: return messagebox.showerror("Input Error", "Invalid hyperparameters.")

        df_unique = self.extracted_df.sort_values('k').reset_index(drop=True)
        grouped = df_unique.groupby('k')
        k_values_sorted = np.array(sorted(grouped.groups.keys()))
        if len(k_values_sorted) == 0: return

        seed_k_actual = k_values_sorted[np.argmin(np.abs(k_values_sorted - seed_k_input))]
        seed_group = grouped.get_group(seed_k_actual)
        seed_row = seed_group.iloc[(seed_group['E'] - seed_E_input).abs().argmin()]
        seed_E = seed_row['E']
        selected_set = [(seed_k_actual, seed_E)]

        def safe_slope(E_new, k_new, E_prev, k_prev): return (E_new - E_prev) / (k_new - k_prev) if (k_new - k_prev) != 0 else 0
        def norm_slope_diff(slope, prev_s): return abs(slope - prev_s) / (abs(prev_s) if abs(prev_s) > 1e-8 else 1)
        def choose_candidate(group, p_k, p_E, p_slope):
            cand = group.copy().reset_index(drop=True)
            cand['slope'] = cand.apply(lambda r: safe_slope(r['E'], r['k'], p_E, p_k), axis=1)
            cand['norm_slope_diff'] = cand['slope'].apply(lambda s: norm_slope_diff(s, p_slope))
            cand['E_diff'] = np.abs(cand['E'] - p_E)
            cand = cand[cand['norm_slope_diff'] <= slope_tol]
            cand = cand[cand['E_diff'] <= E_tol]
            cand = cand.sort_values(['norm_slope_diff', 'E_diff', 'residual'])
            return cand.iloc[0] if not cand.empty else None

        idx_seed = np.where(k_values_sorted == seed_k_actual)[0][0]
        prev_slope = 0.0
        if idx_seed + 1 < len(k_values_sorted):
            try:
                g_next = grouped.get_group(k_values_sorted[idx_seed + 1])
                cand = g_next.loc[g_next['residual'].idxmin()]
                prev_slope = safe_slope(cand['E'], cand['k'], seed_E, seed_k_actual)
            except: pass

        p_k, p_E, p_slope = seed_k_actual, seed_E, prev_slope
        for idx in range(idx_seed + 1, len(k_values_sorted)):
            k_next = k_values_sorted[idx]
            if abs(k_next - p_k) > k_tol: break
            best = choose_candidate(grouped.get_group(k_next), p_k, p_E, p_slope)
            if best is None: continue
            selected_set.append((best['k'], best['E']))
            p_slope = safe_slope(best['E'], best['k'], p_E, p_k); p_k, p_E = best['k'], best['E']

        p_k, p_E, p_slope = seed_k_actual, seed_E, prev_slope
        for idx in range(idx_seed - 1, -1, -1):
            k_next = k_values_sorted[idx]
            if abs(k_next - p_k) > k_tol: break
            best = choose_candidate(grouped.get_group(k_next), p_k, p_E, p_slope)
            if best is None: continue
            selected_set.append((best['k'], best['E']))
            p_slope = safe_slope(best['E'], best['k'], p_E, p_k); p_k, p_E = best['k'], best['E']

        self.selected_points = selected_set
        self.selected_points.sort(key=lambda x: x[0])
        self._update_plot(preserve_limits=True)
        messagebox.showinfo("Auto Select", f"Successfully traced {len(self.selected_points)} points via physical logic.")

    def clear_selection(self):
        self.selected_points, self.spline_func = [], None
        self.btn_next_step.config(state=tk.DISABLED)
        self._update_plot(preserve_limits=True)

    def fit_spline(self):
        if len(self.selected_points) < 4: return messagebox.showwarning("Warning", "Select at least 4 points to fit!")
        try: s_param = float(self.ent_spline_s.get())
        except: return messagebox.showerror("Error", "Invalid Spline s parameter")
            
        pts = np.array(self.selected_points)
        try:
            unique_k, indices = np.unique(pts[:, 0], return_index=True)
            self.spline_func = UnivariateSpline(unique_k, pts[indices, 1], s=s_param)
            self._update_plot(preserve_limits=True)
            self.after(100, lambda: self.btn_next_step.config(state=tk.NORMAL))
        except Exception as e: messagebox.showerror("Spline Fit Error", str(e))

    def go_to_step_2(self):
        if self.controller and hasattr(self.controller, 'notebook'):
            self.controller.notebook.select(self.controller.step2_module)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("ARPES Tool - Step 1 Testing Environment")
    root.geometry("1350x850")
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)
    tab_1_container = ttk.Frame(notebook)
    notebook.add(tab_1_container, text=" Step 1: Preprocessing & Extraction ")
    app_step1 = Step1_BandExtraction(tab_1_container)
    app_step1.pack(fill=tk.BOTH, expand=True)
    root.mainloop()