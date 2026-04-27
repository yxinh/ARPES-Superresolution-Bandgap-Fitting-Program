import tkinter as tk
from tkinter import ttk
from step1_band_extraction import Step1_BandExtraction
from step2_sc_gap_fitting import Step2_GapFitting
from step3_temperature_dependence import Step3_TemperatureDependence

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ARPES Integrated Analysis Suite")
        self.root.geometry("1350x850")
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Band Extraction
        self.step1_module = Step1_BandExtraction(self.notebook, controller=self)
        self.notebook.add(self.step1_module, text=" Step 1: Preprocessing & Extraction ")

        # Tab 2: SC Gap Fitting
        self.step2_module = Step2_GapFitting(self.notebook, controller=self)
        self.notebook.add(self.step2_module, text=" Step 2: SC Gap Fitting & F-Test ")

        # Tab 3: Temperature Dependence (Preparation)
        self.step3_module = Step3_TemperatureDependence(self.notebook, controller=self)
        self.notebook.add(self.step3_module, text=" Step 3: Temperature Dependence ")

if __name__ == "__main__":
    root = tk.Tk()
        
    app = MainApp(root)
    root.mainloop()