# ARPES Integrated Analysis Suite

A comprehensive GUI-based tool developed in Python for the integrated analysis of **Angle-Resolved Photoemission Spectroscopy (ARPES)** data. This suite provides a streamlined workflow from raw data preprocessing to sophisticated physical model fitting.

## 🚀 Core Modules

### 1. Step 1: Preprocessing & Band Extraction
- **Data Loading**: Import raw ARPES data (supports CSV/TSV formats).
- **Background Subtraction**: Integrated **Shirley background** subtraction for noise reduction.
- **ROI Selection**: Interactive Region of Interest (ROI) selection for spectral analysis.
- **Spline Fitting**: Uses `UnivariateSpline` for precise band skeleton extraction from **high-temperature** (normal state) experimental spectra.

### 2. Step 2: SC Gap Fitting & F-Test
- **Advanced Fitting**: Specialized modules for extracting superconducting (SC) gap parameters ($\Delta$).
- **Statistical Analysis**: Implements the **F-Test** to compare different fitting models (e.g., Gap vs. Gapless), ensuring statistical significance in bandgap determination.
- **Physical Modeling**: Accounts for thermal broadening using the Fermi-Dirac distribution and Boltzmann constant ($k_B$).

### 3. Step 3: Temperature Dependence Analysis
- **Scattering Rate ($\Gamma$)**: Automated analysis of the temperature dependence of the scattering rate.
- **Evolution Visualization**: Track how RSS (Residual Sum of Squares) and other physical parameters evolve across different temperature points.
- **Gapless State Detection**: Automatic shading of the normal state based on $T_c$ estimates.

## 🛠️ Requirements

- **Python 3.8+**
- **Tkinter**: GUI framework.
- **NumPy & Pandas**: Data manipulation.
- **Matplotlib**: Publication-quality plotting.
- **SciPy**: Optimization (`curve_fit`), interpolation, and statistical tests.

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/yxinh/ARPES-Superresolution-Bandgap-Fitting-Program.git](https://github.com/yxinh/ARPES-Superresolution-Bandgap-Fitting-Program.git)
   cd ARPES-Superresolution-Bandgap-Fitting-Program
   ```

2. **Create Environment**:
   ```bash
   # Using Conda (Recommended)
   conda env create -f environment.yml
   conda activate arpes-suite
   ```

## 🖥️ Usage

Launch the main application:
```bash
python MainApp.py
```

## 🔄 Workflow

1. **Step 1**: Load and normalize your ARPES data, then extract the band structure from the high-temperature (reference) dataset.
2. **Step 2**: Perform gap fitting on low-temperature data and use the **F-Test** to validate the presence of a spectral gap.
3. **Step 3**: Batch import multiple temperature datasets to analyze the evolution of physics parameters across phase transitions.

## 🎓 Citation

If you use this software in your research or publication, please cite it as follows:

**BibTeX:**
```bibtex
@misc{arpes_suite_2026,
  author = {Yang, Xinhao},
  title = {},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{[https://github.com/yxinh/ARPES-Superresolution-Bandgap-Fitting-Program](https://github.com/yxinh/ARPES-Superresolution-Bandgap-Fitting-Program)}}
}
```

## 📄 License
This project is licensed under the MIT License.

## ✉️ Contact
For questions or collaborations, please open an Issue in this repository.