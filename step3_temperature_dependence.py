import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class Step3_TemperatureDependence(ttk.Frame):
    def __init__(self, parent, controller=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller 
        
        self.temp_data = [] 
        self.extracted_physics = [] 
        self.Tc_estimate = None # 动态记录能隙闭合温度

        self.show_mode = tk.StringVar(value="RSS Comparison vs T")
        
        self._build_ui()

    # =============================================================================
    # --- Publication Ready Style Helper ---
    # =============================================================================
    def _set_scientific_style(self, ax):
        ax.tick_params(direction='in', length=6, width=1.5, colors='k', top=True, right=True, labelsize=12)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.grid(True, linestyle='--', alpha=0.6)

    def _apply_gapless_shading(self, ax, T_max):
        """在图表上绘制代表 Gapless (Normal State) 的灰色区域"""
        if self.Tc_estimate is not None and self.Tc_estimate <= T_max:
            ax.axvspan(self.Tc_estimate, T_max + 50, color='gray', alpha=0.3, label='Gapless Region')

    # =============================================================================
    # --- UI Building ---
    # =============================================================================
    def _build_ui(self):
        self.left_panel = ttk.Frame(self, width=320)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.left_panel.pack_propagate(False)

        self.right_panel = ttk.Frame(self)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._build_left_panel()
        self._build_plot_area()

    def _build_left_panel(self):
        # 1. Data Loading
        frame_load = ttk.LabelFrame(self.left_panel, text="1. Load Data", padding=5)
        frame_load.pack(fill=tk.X, pady=5)
        ttk.Button(frame_load, text="Load from Directory (Files)", command=self.load_from_folder).pack(fill=tk.X, pady=2)
        ttk.Button(frame_load, text="Load from Step 2 (Memory)", command=self.load_from_step2).pack(fill=tk.X, pady=2)
        self.lbl_status = ttk.Label(frame_load, text="Status: No data loaded.", foreground="blue")
        self.lbl_status.pack(fill=tk.X, pady=2)

        # 2. Analysis Parameters
        frame_param = ttk.LabelFrame(self.left_panel, text="2. Analysis Parameters", padding=5)
        frame_param.pack(fill=tk.X, pady=5)
        
        f1 = ttk.Frame(frame_param); f1.pack(fill=tk.X, pady=2)
        ttk.Label(f1, text="P-value Threshold:").pack(side=tk.LEFT)
        self.ent_p_thresh = ttk.Entry(f1, width=8); self.ent_p_thresh.insert(0, "0.05"); self.ent_p_thresh.pack(side=tk.RIGHT)
        
        f2 = ttk.Frame(frame_param); f2.pack(fill=tk.X, pady=2)
        ttk.Label(f2, text="Target k (or 'kF'):").pack(side=tk.LEFT)
        self.ent_k_ref = ttk.Entry(f2, width=8); self.ent_k_ref.insert(0, "kF"); self.ent_k_ref.pack(side=tk.RIGHT)

        f3 = ttk.Frame(frame_param); f3.pack(fill=tk.X, pady=2)
        ttk.Label(f3, text="Err Mult for Weighting:").pack(side=tk.LEFT)
        self.ent_err_mult = ttk.Entry(f3, width=8); self.ent_err_mult.insert(0, "2.0"); self.ent_err_mult.pack(side=tk.RIGHT)
        
        ttk.Button(frame_param, text="Recalculate Physics", command=self._calculate_physics).pack(fill=tk.X, pady=5)

        # 3. Visualization Mode
        frame_vis = ttk.LabelFrame(self.left_panel, text="3. Visualization Mode", padding=5)
        frame_vis.pack(fill=tk.X, pady=5)
        modes = ["RSS Comparison vs T", "P-value vs T", "SC Gap (Delta) vs T", "Gamma vs T"]
        for m in modes:
            ttk.Radiobutton(frame_vis, text=m, variable=self.show_mode, value=m, command=self._update_plot).pack(anchor=tk.W, pady=2)

        # 4. Display Range Override
        frame_lim = ttk.LabelFrame(self.left_panel, text="4. Display Ranges", padding=5)
        frame_lim.pack(fill=tk.X, pady=5)
        
        # Temp limits
        f_t = ttk.Frame(frame_lim); f_t.pack(fill=tk.X, pady=2)
        ttk.Label(f_t, text="Temp (K):", width=12).pack(side=tk.LEFT)
        self.ent_t_min = ttk.Entry(f_t, width=6); self.ent_t_min.insert(0, "auto"); self.ent_t_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(f_t, text="to").pack(side=tk.LEFT); 
        self.ent_t_max = ttk.Entry(f_t, width=6); self.ent_t_max.insert(0, "auto"); self.ent_t_max.pack(side=tk.LEFT, padx=1)

        # Delta limits
        f_d = ttk.Frame(frame_lim); f_d.pack(fill=tk.X, pady=2)
        ttk.Label(f_d, text="Delta (meV):", width=12).pack(side=tk.LEFT)
        self.ent_d_min = ttk.Entry(f_d, width=6); self.ent_d_min.insert(0, "auto"); self.ent_d_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(f_d, text="to").pack(side=tk.LEFT); 
        self.ent_d_max = ttk.Entry(f_d, width=6); self.ent_d_max.insert(0, "auto"); self.ent_d_max.pack(side=tk.LEFT, padx=1)

        # Gamma limits
        f_g = ttk.Frame(frame_lim); f_g.pack(fill=tk.X, pady=2)
        ttk.Label(f_g, text="Gamma (meV):", width=12).pack(side=tk.LEFT)
        self.ent_g_min = ttk.Entry(f_g, width=6); self.ent_g_min.insert(0, "auto"); self.ent_g_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(f_g, text="to").pack(side=tk.LEFT); 
        self.ent_g_max = ttk.Entry(f_g, width=6); self.ent_g_max.insert(0, "auto"); self.ent_g_max.pack(side=tk.LEFT, padx=1)

        ttk.Button(frame_lim, text="Apply Limits", command=self._update_plot).pack(fill=tk.X, pady=5)

    def _build_plot_area(self):
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right_panel)
        self.toolbar.update()

    # =============================================================================
    # --- Data Loading (Updated with kF Header Parsing) ---
    # =============================================================================
    def load_from_folder(self):
        folder_path = filedialog.askdirectory(title="Select Step 2 Output Directory")
        if not folder_path: return
        
        try:
            self.temp_data = []
            files = [f for f in os.listdir(folder_path) if f.startswith('fit_results_') and f.endswith('.txt')]
            if not files: return messagebox.showerror("Error", "No 'fit_results_*.txt' files found.")
                
            for fname in files:
                fpath = os.path.join(folder_path, fname)
                
                T_val, kF_val = None, None
                
                # 读取并解析文件头部，获取 T 和 kF
                with open(fpath, 'r') as f:
                    lines = [f.readline() for _ in range(3)]
                    # 第二行通常是: Temperature: 14.0 K, kF: 0.1234
                    if len(lines) >= 2 and 'Temperature' in lines[1]:
                        parts = lines[1].split(',')
                        for p in parts:
                            if 'Temperature' in p:
                                try: T_val = float(p.split(':')[1].replace('K', '').strip())
                                except: pass
                            if 'kF' in p:
                                try: kF_val = float(p.split(':')[1].strip())
                                except: pass
                
                # 如果没读到 T，使用文件名 fallback
                if T_val is None:
                    T_val = float(fname.replace('fit_results_', '').replace('K.txt', ''))
                
                # 读取矩阵数据
                data = np.loadtxt(fpath, skiprows=4)
                if data.shape[1] < 8: continue
                
                self.temp_data.append({
                    'T': T_val, 'kF': kF_val,
                    'k_vals': data[:, 0], 'delta_vals': data[:, 1], 'err_vals': data[:, 2],
                    'gamma_vals': data[:, 3], 'gamma_err_vals': data[:, 4],
                    'RSS_gap': data[:, 5], 'RSS_met': data[:, 6], 'p_vals': data[:, 7]
                })
            
            self.temp_data.sort(key=lambda x: x['T'])
            self.lbl_status.config(text=f"Loaded {len(self.temp_data)} files.", foreground="green")
            self._calculate_physics()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def load_from_step2(self):
        if not self.controller or not hasattr(self.controller, 'step2_module'): return
        step2 = self.controller.step2_module
        if not hasattr(step2, 'saved_results') or not step2.saved_results: return
            
        try:
            self.temp_data = []
            for key, res in step2.saved_results.items():
                stats = res.get('final_stats', {})
                self.temp_data.append({
                    'T': res.get('Temperature', 0),
                    'kF': res.get('kF', None),  # 直接从内存加载保存的 kF
                    'k_vals': np.array(res.get('k_points', [])),
                    'delta_vals': np.array(stats.get('delta_fit', [])),
                    'err_vals': np.array(stats.get('delta_err', [])),
                    'gamma_vals': np.array(stats.get('gamma_fit', [])),
                    'gamma_err_vals': np.array(stats.get('gamma_err', [])),
                    'RSS_gap': np.array(stats.get('RSS_gap', [])),
                    'RSS_met': np.array(stats.get('RSS_met', [])),
                    'p_vals': np.array(stats.get('p_vals', []))
                })
            self.temp_data.sort(key=lambda x: x['T'])
            self.lbl_status.config(text=f"Loaded {len(self.temp_data)} sets (Memory).", foreground="green")
            self._calculate_physics()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    # =============================================================================
    # --- Physics Calculation (Single Point & Dynamic Weighted) ---
    # =============================================================================
    def _calculate_physics(self):
        if not self.temp_data: return
        
        try:
            p_thresh = float(self.ent_p_thresh.get())
            k_ref_str = self.ent_k_ref.get()
            err_mult = float(self.ent_err_mult.get())
        except ValueError:
            return messagebox.showerror("Error", "Invalid analysis parameters.")
            
        self.extracted_physics = []
        self.Tc_estimate = None
        gap_closed = False
        
        for item in self.temp_data:
            T = item['T']
            kF_val = item['kF']
            k_vals, p_vals = item['k_vals'], item['p_vals']
            
            # Determine Target k (优先使用真实的 kF)
            if k_ref_str.lower() == 'kf':
                if kF_val is not None:
                    k_target = kF_val
                else:
                    k_target = k_vals[len(k_vals) // 2] # Fallback 旧版逻辑
            else:
                k_target = float(k_ref_str)
                
            target_idx = np.argmin(np.abs(k_vals - k_target))
            
            # 1. Single Point Method
            sp_delta = item['delta_vals'][target_idx]
            sp_err = item['err_vals'][target_idx]
            sp_gamma = item['gamma_vals'][target_idx]
            sp_g_err = item['gamma_err_vals'][target_idx]
            sp_p_val = p_vals[target_idx]
            
            # Estimate Tc
            if sp_p_val > p_thresh and not gap_closed:
                self.Tc_estimate = T
                gap_closed = True
                
            # 2. Dynamic Weighted Method (基于误差倍数的左右拓展搜索)
            # --- For Delta ---
            lb_d = sp_delta - err_mult * sp_err
            ub_d = sp_delta + err_mult * sp_err
            
            left_d = target_idx
            while left_d > 0 and lb_d <= item['delta_vals'][left_d-1] <= ub_d: left_d -= 1
            right_d = target_idx
            while right_d < len(k_vals)-1 and lb_d <= item['delta_vals'][right_d+1] <= ub_d: right_d += 1
            
            valid_idx_d = list(range(left_d, right_d+1))
            w_d = 1.0 / (item['err_vals'][valid_idx_d]**2 + 1e-15)
            w_delta = np.sum(w_d * item['delta_vals'][valid_idx_d]) / np.sum(w_d)
            w_err = np.sqrt(1.0 / np.sum(w_d))

            # --- For Gamma ---
            lb_g = sp_gamma - err_mult * sp_g_err
            ub_g = sp_gamma + err_mult * sp_g_err
            
            left_g = target_idx
            while left_g > 0 and lb_g <= item['gamma_vals'][left_g-1] <= ub_g: left_g -= 1
            right_g = target_idx
            while right_g < len(k_vals)-1 and lb_g <= item['gamma_vals'][right_g+1] <= ub_g: right_g += 1
            
            valid_idx_g = list(range(left_g, right_g+1))
            w_g = 1.0 / (item['gamma_err_vals'][valid_idx_g]**2 + 1e-15)
            w_gamma = np.sum(w_g * item['gamma_vals'][valid_idx_g]) / np.sum(w_g)
            w_g_err = np.sqrt(1.0 / np.sum(w_g))
                
            self.extracted_physics.append({
                'T': T, 'k_target': k_target,
                'sp_p_val': sp_p_val,
                'rss_gap_mean': np.mean(item['RSS_gap']),
                'rss_met_mean': np.mean(item['RSS_met']),
                'sp_delta': sp_delta, 'sp_err': sp_err,
                'w_delta': w_delta, 'w_err': w_err,
                'sp_gamma': sp_gamma, 'sp_g_err': sp_g_err,
                'w_gamma': w_gamma, 'w_g_err': w_g_err
            })
            
        self._update_plot()

    # =============================================================================
    # --- Plotting Logic ---
    # =============================================================================
    def _update_plot(self):
        if not self.extracted_physics: return
        self.fig.clf()
        
        mode = self.show_mode.get()
        T_arr = np.array([p['T'] for p in self.extracted_physics])
        T_max = np.max(T_arr)
        
        ax = self.fig.add_subplot(111)
        
        # 1. 绘制独立的 RSS 比较图
        if mode == "RSS Comparison vs T":
            rss_g = [p['rss_gap_mean'] for p in self.extracted_physics]
            rss_m = [p['rss_met_mean'] for p in self.extracted_physics]
            
            ax.plot(T_arr, rss_g, '-o', color=[0, 0.2, 0.6], markerfacecolor=[0, 0.2, 0.6], 
                     linewidth=1.8, label='RSS: Gap Model')
            ax.plot(T_arr, rss_m, '--s', color=[0.7, 0, 0], markerfacecolor=[0.7, 0, 0], 
                     linewidth=1.8, label='RSS: Zero Gap Model')
            ax.set_xlabel('Temperature $T$ (K)', fontsize=16)
            ax.set_ylabel('Residual Sum of Squares (RSS)', fontsize=16)
            ax.set_title('Model Comparison', fontsize=18)
            self._apply_gapless_shading(ax, T_max)
            ax.legend(loc='best', fontsize=12)
            self._set_scientific_style(ax)

        # 2. 绘制独立的 P-value 图
        elif mode == "P-value vs T":
            p_vals = [p['sp_p_val'] for p in self.extracted_physics]
            log_p = np.log10(np.clip(p_vals, 1e-15, 1))
            
            ax.plot(T_arr, log_p, '-D', color=[0.3, 0, 0.5], markerfacecolor=[0.3, 0, 0.5], 
                     linewidth=1.8, markersize=8)
            
            try:
                thresh = float(self.ent_p_thresh.get())
                ax.axhline(np.log10(thresh), color='k', linestyle=':', linewidth=2, label=f'Threshold (p = {thresh})')
            except: pass
            
            ax.set_xlabel('Temperature $T$ (K)', fontsize=16)
            ax.set_ylabel(r'$\log_{10}(P$-value$)$', fontsize=16)
            ax.set_title('Significance of Superconducting Gap', fontsize=18)
            self._apply_gapless_shading(ax, T_max)
            ax.legend(loc='best', fontsize=12)
            self._set_scientific_style(ax)
            
        # 3. 绘制 SC Gap 图
        elif mode == "SC Gap (Delta) vs T":
            sp_d = np.array([p['sp_delta'] for p in self.extracted_physics]) * 1000
            sp_e = np.array([p['sp_err'] for p in self.extracted_physics]) * 1000
            w_d = np.array([p['w_delta'] for p in self.extracted_physics]) * 1000
            w_e = np.array([p['w_err'] for p in self.extracted_physics]) * 1000
            
            ax.errorbar(T_arr, sp_d, yerr=sp_e, fmt='-o', color=[0, 0.4470, 0.7410], 
                        markerfacecolor=[0, 0.4470, 0.7410], markeredgecolor='k',
                        linewidth=2, capsize=4, label='Single Point (at target $k_F$)')
            
            ax.errorbar(T_arr, w_d, yerr=w_e, fmt='--s', color=[0.8500, 0.3250, 0.0980], 
                        markerfacecolor=[0.8500, 0.3250, 0.0980], markeredgecolor='k',
                        linewidth=2, capsize=4, label='Inverse-Variance Weighted')
            
            ax.set_xlabel('Temperature $T$ (K)', fontsize=16)
            ax.set_ylabel(r'Superconducting Gap $\Delta$ (meV)', fontsize=16)
            ax.set_title(r'Temperature Dependence of SC Gap', fontsize=18)
            self._apply_gapless_shading(ax, T_max)
            ax.legend(loc='best', fontsize=12)
            self._set_scientific_style(ax)
            
            # Apply Y Limits
            try: ymin = float(self.ent_d_min.get()) if self.ent_d_min.get() != 'auto' else None
            except: ymin = None
            try: ymax = float(self.ent_d_max.get()) if self.ent_d_max.get() != 'auto' else None
            except: ymax = None
            ax.set_ylim(bottom=ymin, top=ymax)

        # 4. 绘制 Gamma 图
        elif mode == "Gamma vs T":
            sp_g = np.array([p['sp_gamma'] for p in self.extracted_physics]) * 1000
            sp_ge = np.array([p['sp_g_err'] for p in self.extracted_physics]) * 1000
            w_g = np.array([p['w_gamma'] for p in self.extracted_physics]) * 1000
            w_ge = np.array([p['w_g_err'] for p in self.extracted_physics]) * 1000
            
            ax.errorbar(T_arr, sp_g, yerr=sp_ge, fmt='-o', color=[0, 0.4470, 0.7410], 
                        markerfacecolor=[0, 0.4470, 0.7410], markeredgecolor='k',
                        linewidth=2, capsize=4, label='Single Point (at target $k_F$)')
            
            ax.errorbar(T_arr, w_g, yerr=w_ge, fmt='--s', color=[0.4660, 0.6740, 0.1880], 
                        markerfacecolor=[0.4660, 0.6740, 0.1880], markeredgecolor='k',
                        linewidth=2, capsize=4, label='Inverse-Variance Weighted')
            
            ax.set_xlabel('Temperature $T$ (K)', fontsize=16)
            ax.set_ylabel(r'Scattering Rate $\Gamma$ (meV)', fontsize=16)
            ax.set_title(r'Temperature Dependence of Scattering Rate', fontsize=18)
            self._apply_gapless_shading(ax, T_max)
            ax.legend(loc='best', fontsize=12)
            self._set_scientific_style(ax)
            
            # Apply Y Limits
            try: ymin = float(self.ent_g_min.get()) if self.ent_g_min.get() != 'auto' else None
            except: ymin = None
            try: ymax = float(self.ent_g_max.get()) if self.ent_g_max.get() != 'auto' else None
            except: ymax = None
            ax.set_ylim(bottom=ymin, top=ymax)
            
        # Apply X (Temperature) Limits across all modes
        try: tmin = float(self.ent_t_min.get()) if self.ent_t_min.get() != 'auto' else None
        except: tmin = None
        try: tmax = float(self.ent_t_max.get()) if self.ent_t_max.get() != 'auto' else None
        except: tmax = None
        if tmin is not None or tmax is not None:
            ax.set_xlim(left=tmin, right=tmax)
        else:
            ax.set_xlim(left=np.min(T_arr) - 2, right=T_max + 10)
                
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Step 3 Test")
    root.geometry("1000x700")
    app = Step3_TemperatureDependence(root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()