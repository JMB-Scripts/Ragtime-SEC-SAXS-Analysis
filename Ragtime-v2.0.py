import tkinter as tk
from tkinter import messagebox, filedialog,ttk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import linregress
from scipy import integrate
from scipy.integrate import trapezoid
from scipy.integrate import simpson  
import os

class SAXSAnalysisApp:
    def __init__(self, root):
        # Initialize the main window
        self.root = root
        self.root.title("Ragtime Sec-SAXS Analysis")

        # Store file data
        self.files_data = []  # List to hold file data in the format (filename, data)
        
        #initalisation
        self.best_file = None
        self.data = None
        self.MW = None # initialize MW here
        self.qmin_entry = tk.Entry(root)  # Entry widget for qmin
        self.qmax_entry = tk.Entry(root)  # Entry widget for qmax
        #create the label first.
        self.qmax_rg = tk.Label(root, text="qmax*Rg: ")
        self.qmin_rg = tk.Label(root, text="qmin*Rg: ")

    # GUI Elements (using grid)
        self.browse_button = tk.Button(root, text="Browse", command=self.browse_directory)
        self.browse_button.grid(row=0, column=0, columnspan=2, sticky="ew")

        self.folder_label = tk.Label(self.root, text="Selected Folder: None", fg="blue")
        self.folder_label.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.file_label = tk.Label(root, text="No file selected", fg="black")
        self.file_label.grid(row=2, column=0, columnspan=2, sticky="ew")

        self.qmin_label = tk.Label(root, text="Enter qmin (line number):")
        self.qmin_label.grid(row=3, column=0)
        self.qmin_entry = tk.Entry(root)
        self.qmin_entry.grid(row=3, column=1)
        self.qmax_label = tk.Label(root, text="Enter qmax (line number):")
        self.qmax_label.grid(row=4, column=0)
        self.qmax_entry = tk.Entry(root)
        self.qmax_entry.grid(row=4, column=1)

        self.rg_label = tk.Label(root, text="Rg: ")
        self.rg_label.grid(row=3, column=2, sticky="w")

        self.i0_label = tk.Label(root, text="I(0): ")
        self.i0_label.grid(row=4, column=2, sticky="w")

        self.qmin_rg_label = tk.Label(root, text="qmin*Rg: ")
        self.qmin_rg.grid(row=5, column=2, sticky="w")
        self.qmax_rg = tk.Label(root, text="qmax*Rg: ")
        self.qmax_rg.grid(row=6, column=2, sticky="w")

        self.mw_label = tk.Label(root, text="MW: ")
        self.mw_label.grid(row=7, column=2, sticky="w")

        self.auto_guinier_button = tk.Button(root, text="Auto-Guinier", command=self.auto_guinier_analysis)
        self.auto_guinier_button.grid(row=6, column=0, columnspan=2, sticky="ew")

        self.manual_guinier_button = tk.Button(root, text="Manual-Guinier", command=self.manual_guinier_analysis)
        self.manual_guinier_button.grid(row=7, column=0, columnspan=2, sticky="ew")

        self.apply_to_all_button = tk.Button(root, text="Ragtime", command=self.apply_to_all)
        self.apply_to_all_button.grid(row=8, column=0, columnspan=2, sticky="ew")
        # Add Reset button
        self.reset_button = tk.Button(root, text="Reset", command=self.reset)
        self.reset_button.grid(row=9, column=0, columnspan=2, sticky="ew")

        self.quit_button = tk.Button(root, text="Quit", command=self.quit)
        self.quit_button.grid(row=10, column=0, columnspan=2, sticky="ew")
        
        # Add version label
        version = "JMB-Scripts - Ragtime Sec-SAXS Analysis - v2.0 -"  # Replace with your actual version
        self.version_label = tk.Label(root, text=version, fg="gray", font=("TkDefaultFont", 12))
        self.version_label.grid(row=11, column=0, columnspan=2, sticky="ew")



################################
#
# BROWSE
#
#################################

    def browse_directory(self):
        """Browse for SAXS data files and determine the best file based on highest intensity at q=0.1."""
        folder_path = filedialog.askdirectory()
        
        if not folder_path:
            return
        
        self.directory_path = folder_path #store directory path

        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".dat")])
        if not files:
            messagebox.showerror("Error", "No .dat files found in the selected directory.")
            return
        
        # Create progress bar window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing Files...")
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
        progress_bar.pack(padx=10, pady=10)
        
        #load all files
        self.load_files(folder_path, progress_bar, len(files)) #load all files with progress

        # Update GUI with the chosen folder path
        self.folder_label.config(text=f"Selected Folder: {folder_path}", fg="blue")


        """ find best file"""
        best_intensity, best_file, best_data = -np.inf, None, None

        for i, file in enumerate(files):
            file_path = os.path.join(folder_path, file)
            data = self.read_valid_data(file_path)
            if data is None:
                continue

            q, I = data[:, 0], data[:, 1]
            idx = np.argmin(np.abs(q - 0.1))  # Find closest q=0.1
            intensity = I[idx]

            if intensity > best_intensity:
                best_intensity = intensity
                best_file = file_path
                best_data = data

            # Update progress bar
            progress_bar["value"] = (i + 1) / len(files) * 100
            progress_window.update_idletasks()

        if best_file:
            self.best_file = best_file
            self.data = best_data
            self.file_label.config(text=f"Best file: {os.path.basename(best_file)}", fg="red")

        
        # Destroy progress bar window
        progress_window.destroy()

################################
#
# load files 
#
################################# 

    def load_files(self, directory, progress_bar, total_files):
        """Load all .dat files from the selected directory and sort them with progress bar."""
        self.files_data = []
        files = sorted(os.listdir(directory))
        for i, filename in enumerate(files):
            if filename.endswith(".dat"):
                file_path = os.path.join(directory, filename)
                data = self.read_valid_data(file_path)
                if data is not None:
                    self.files_data.append((filename, data))

            # Update progress bar
            progress_bar["value"] = (i + 1) / total_files * 100
            progress_bar.master.update_idletasks()
        if not self.files_data:
            messagebox.showerror("Error", "No valid .dat files found in the selected directory.")
        #else:
            #messagebox.showinfo("Files Loaded", f"{len(self.files_data)} files loaded successfully.")

################################
#
# Remove the header 
#
#################################

    def read_valid_data(self, file_path):
        """Reads a .dat file and extracts valid SAXS data with three numeric columns (q, I, err)."""
        try:
            data = []
            with open(file_path, "r") as f:
                for line in f:
                    # Ignore empty lines and comment lines (starting with #)
                    if not line.strip() or line.startswith("#"):
                        continue
                    
                    parts = line.strip().split()
                    
                    # Ensure the line contains exactly 3 numeric values
                    if len(parts) >= 3:
                        try:
                            q, I, err = map(float, parts)
                            # Check if any of the values are NaN and skip the line if so
                            if np.isnan(q) or np.isnan(I) or np.isnan(err):
                                continue
                            data.append([q, I, err])
                        except ValueError:
                            continue  # Skip lines that contain invalid numbers
            
            if not data:
                messagebox.showwarning("Warning", f"No valid SAXS data found in {file_path}.")
                return None  # Return None if no valid data is read

            return np.array(data, dtype=float)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read {file_path}: {e}")
            return None
        
################################
#
# Auto guinier on best file
#
#################################        

    def auto_guinier_analysis(self):
        """Auto-detect the Guinier region using a sliding window method."""
        if self.best_file is None or self.data is None:
            messagebox.showerror("Error", "No best file selected.")
            return

        q, I = self.data[:, 0], self.data[:, 1]
        q_squared, ln_I = q**2, np.log(I)

        best_score, best_qmin_idx, best_qmax_idx = -np.inf, 0, 0
        best_rg, best_i0, best_r2 = 0, 0, 0

        for window_size in range(5, min(160, len(q_squared))):
            for start_idx in range(0, min(40, len(q_squared) - window_size)):
                end_idx = start_idx + window_size

                # Linear regression
                slope, intercept, r_value, _, std_err = linregress(q_squared[start_idx:end_idx], ln_I[start_idx:end_idx])
                r2 = r_value**2

                if slope >= 0:  # Ignore non-physical results
                    continue

                rg = np.sqrt(-3 * slope)
                i0 = np.exp(intercept)
                qmin_rg, qmax_rg = q[start_idx] * rg, q[end_idx - 1] * rg

                if not (1.1 <= qmax_rg <= 1.5) or qmin_rg > 0.8:
                    continue

                # Scoring criteria
                score = (50 if qmin_rg < 0.3 else 40 * (1 - qmin_rg/0.8)) + \
                        40 * (1 - abs(qmax_rg - 1.3)/0.4) + \
                        50 * (r2**4)

                if score > best_score:
                    best_score = score
                    best_qmin_idx, best_qmax_idx = start_idx, end_idx - 1
                    best_rg, best_i0, best_r2 = rg, i0, r2

        if best_score > -np.inf:
            self.qmin_entry.delete(0, tk.END)
            self.qmin_entry.insert(0, str(best_qmin_idx))
            self.qmax_entry.delete(0, tk.END)
            self.qmax_entry.insert(0, str(best_qmax_idx))
        # Calculate MW using the estimate_molecular_weight function
            MW = self.estimate_molecular_weight(q, I,best_i0, best_rg)
            
        # Generate 4-panel plot after Auto-Guinier
            self.generate_4_panel_plot(q, I, best_qmin_idx, best_qmax_idx, best_rg, best_i0)


        else:
            messagebox.showwarning("Estimation Failed", "Could not determine a valid Guinier region.")

################################
#
# MW 
#
################################# 

    def estimate_molecular_weight(self, q, I, best_i0, best_rg):
        """Estimate MW using VC method (integration up to q=0.3)."""
        # Prepare VC plot
        q_filtered = q[q <= 0.3]
        I_filtered = I[q <= 0.3]
        yvc = I_filtered * q_filtered
        Intgr = integrate.simpson(yvc,x=q_filtered)
        VC = best_i0 / Intgr
        QR = VC**2 / best_rg
        self.MW = QR / 0.1231  # Store MW
        return self.MW

################################
#
# 4 Panels Plots for auto-guinier and manual-guinier
#
#################################    

    def generate_4_panel_plot(self, q, I, qmin_idx, qmax_idx, rg, i0):
        """Generate the 4-panel plot for the Guinier analysis."""
        #couleur
        col1='#0D92F4' #blue
        col2='#C62E2E' #red
        
        # Prepare plots:
        q_squared = q**2
        ln_I = np.log(I)
        
        # Data points for fitting
        x_data = q_squared[qmin_idx:qmax_idx+1]
        y_data = ln_I[qmin_idx:qmax_idx+1]

        # Calculate model values at the data points
        y_model = -rg**2 / 3 * x_data + np.log(i0)

        # Compute residuals
        residuals = y_data - y_model

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ####################
        # 1st panel: I(q) vs q

        axes[0, 0].plot(q, np.log(I), color=col1, markersize=3)
        axes[0, 0].set_xlabel('q (Å⁻¹)')
        axes[0, 0].set_ylabel('ln(I(q))')
        axes[0, 0].set_title('I(q) vs q')
       
        ####################
        # 2nd panel: Guinier plot (q² vs ln(I))

        axes[1, 0].plot(q_squared[qmin_idx:qmax_idx+1], ln_I[qmin_idx:qmax_idx+1], 'o', color=col1)
        axes[1, 0].set_xlabel('q² (Å⁻²)')
        axes[1, 0].set_ylabel('ln(I(q))')
        axes[1, 0].set_title('Guinier Plot (q² vs ln(I))')

        # Add fitted line
        x_fit = np.linspace(q_squared[qmin_idx], q_squared[qmax_idx], 100)
        y_fit = -rg**2 / 3 * x_fit + np.log(i0)
        axes[1, 0].plot(x_fit, y_fit, color=col2)

        # Add legend with I(0), qmin·Rg, and qmax·Rg
        legend_text = (f'Rg = {rg:.2f}\n'
                    f'I(0) = {i0:.2f}\n'
                    f'qmin·Rg = {q[qmin_idx] * rg:.2f}\n'
                    f'qmax·Rg = {q[qmax_idx] * rg:.2f}\n'
                    f'qmin_offset: {qmin_idx:.0f}\n'
                    f'qmax_offset: {qmax_idx:.0f}'
                    )
        axes[1, 0].legend(title=legend_text, loc='upper right', fontsize=8)

        # Add inset for residuals
        ax_inset = inset_axes(axes[1, 0], width="40%", height="20%", loc="lower left")
        ax_inset.plot(x_data, residuals,'o',color=col1, markersize=3)
        ax_inset.axhline(0, linestyle='-', color='red', alpha=0.7)
        ax_inset.set_title('Residuals', fontsize=6)
        ax_inset.set_xlabel(r'$q^2$', fontsize=6)
        ax_inset.set_ylabel('Residual', fontsize=6)
        ax_inset.tick_params(axis='both', labelsize=6)
        ax_inset.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')
        axes[1, 0].grid(True)

        ####################
        # 3rd panel: Rg-normalized Kratky plot

        axes[0, 1].plot(q * rg, (q * rg)**2 * I / i0, color=col1, markersize=3)
        axes[0, 1].set_xlabel('q * Rg (Å)')
        axes[0, 1].set_ylabel('qRg² * I(q) / I(0)')
        axes[0, 1].set_title('Rg-normalized Kratky Plot')
        axes[0, 1].axvline(x=1.73, color='grey', linestyle='--')
        axes[0, 1].axhline(y=1.1, color='grey', linestyle='--')
        axes[0, 1].grid(True)

        ####################
        # 4th panel: I*q vs q and Integral of I*q vs q used to calculate MW
        q_limit_idx = np.searchsorted(q, 0.3)  # Find first index where q >= 0.3

        # Compute cumulative integral curve using integrate.simpson
        cumulative_integral = np.array([
            integrate.simpson(q[:i+1] * I[:i+1], x=q[:i+1]) if i > 0 else 0 
            for i in range(1, q_limit_idx)
        ])

        # Plot VC q * I(q) vs q
        axes[1, 1].plot(q[:q_limit_idx], q[:q_limit_idx] * I[:q_limit_idx], color=col1, label='q * I(q)')
        axes[1, 1].set_xlabel('q (Å⁻¹)')
        axes[1, 1].set_ylabel('q * I(q)', color=col1)
        axes[1, 1].tick_params(axis='y', labelcolor=col1)

        # Create a secondary y-axis for the integral curve
        # Format MW text for legend
        mw_text = f"MW: {self.MW:.0f}" if self.MW is not None else "MW: Not calculated"
        ax2 = axes[1, 1].twinx()
        ax2.plot(q[1:q_limit_idx], cumulative_integral, color=col2, label=f'Integral (q ≤ 0.3)\n{mw_text}')
        ax2.set_ylabel('Integrated Intensity', color=col2)
        ax2.tick_params(axis='y', labelcolor=col2)

        # Add title and legends
        axes[1, 1].set_title('VC Plot for MW Estimation')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='center right')

        ####################
        # Update gui

        qmin_rg = q[qmin_idx] * rg 
        qmax_rg = q[qmax_idx] * rg 
        self.update_results(rg, i0, self.MW, qmin_rg,qmax_rg)
        
        ####################    
        # save plot
        plot_path_png = os.path.join(self.directory_path, "Sexier-SAXS.png")
        plot_path_svg = os.path.join(self.directory_path, "Sexier-SAXS.svg")
        plt.savefig(plot_path_png)
        plt.savefig(plot_path_svg, format="svg") 
        
        ####################
        # Display the plot
        
        plt.tight_layout()
        plt.show()

################################
#
# Manual guinier using value in the GUI
#
#################################   

    def manual_guinier_analysis(self):
        if self.best_file is None or self.data is None:
            messagebox.showerror("Error", "No best file selected.")
            return  # Return should be inside the if block

        try:  # This and all following lines should be indented under the method
            qmin_idx, qmax_idx = int(self.qmin_entry.get()), int(self.qmax_entry.get())
            q, I = self.data[:, 0], self.data[:, 1]
            q_squared, ln_I = q**2, np.log(I)

            # Perform Guinier analysis (linear regression for q^2 vs ln(I))
            slope, intercept, r_value, _, std_err = linregress(q_squared[qmin_idx:qmax_idx+1], ln_I[qmin_idx:qmax_idx+1])
            best_rg, best_i0 = np.sqrt(-3 * slope), np.exp(intercept)
            
            #calculation of qmin rg and qmaxrg
            qmin_rg, qmax_rg = q[qmin_idx] * best_rg, q[qmax_idx] * best_rg
        
        # Calculate MW using the estimate_molecular_weight function
            MW = self.estimate_molecular_weight(q, I,best_i0, best_rg)
                               
            # Generate 4-panel plot after Auto-Guinier
            best_qmin_idx = qmin_idx
            best_qmax_idx = qmax_idx
            self.generate_4_panel_plot(q, I, best_qmin_idx, best_qmax_idx, best_rg, best_i0)
            self.update_results(best_rg, best_i0, self.MW, qmin_rg, qmax_rg)
    
        except Exception as e:
            messagebox.showerror("Error", str(e))
################################
#
# Create a fonction to update the GUI after guinier
#
################################# 

    def update_results(self, best_rg, best_i0, MW, qmin_rg, qmax_rg):
        """Update the result labels with new values."""
        self.rg_label.config(text=f"Rg: {best_rg:.2f} Å")
        self.i0_label.config(text=f"I(0): {best_i0:.2f}")
        self.qmin_rg.config(text=f"qmin*Rg: {qmin_rg:.2f}")
        self.qmax_rg.config(text=f"qmax*Rg: {qmax_rg:.2f}")
        self.mw_label.config(text=f"MW: {self.MW:.0f}")
 
################################
#
# Ragtime to all files
#
################################# 

    def apply_to_all(self):
        """Apply the analysis to all files in the dataset."""
        if not self.files_data:
            messagebox.showerror("Error", "No files loaded.")
            return

        try:
            qmin_idx = int(self.qmin_entry.get())
            qmax_idx = int(self.qmax_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid values for qmin and qmax.")
            return

        i0_values, rg_values, mw_values, file_numbers = [], [], [], [] #add file_numbers list
        for file_index, (file_name, data) in enumerate(self.files_data): #add file_index
            q, I = data[:, 0], data[:, 1]
            q_squared, ln_I = q**2, np.log(I)

            try:
                slope, intercept, _, _, std_err = linregress(q_squared[qmin_idx:qmax_idx + 1], ln_I[qmin_idx:qmax_idx + 1])
                rg = np.sqrt(-3 * slope)
                i0 = np.exp(intercept)
                mw = self.estimate_molecular_weight(q, I, i0, rg)
                i0_values.append(i0)
                rg_values.append(rg)
                mw_values.append(mw)
                file_numbers.append(file_index + 1) #file_index +1 to start at 1
            except Exception as e:
                messagebox.showerror("Error", f"Analysis failed for file {file_name}: {e}")
                i0_values.append(np.nan)
                rg_values.append(np.nan)
                mw_values.append(np.nan)
                file_numbers.append(file_index + 1)

        self.generate_2_panel_plot(i0_values, rg_values, mw_values, file_numbers,qmin_idx,qmax_idx)


################################
#
# 2 panel plots:
#  I(0),Rg vs file number 
#  I(0) Mw vs vs file number
#
################################# 

 
    def generate_2_panel_plot(self, i0_values, rg_values, mw_values, file_numbers,qmin_idx,qmax_idx): #add file_numbers
        """Create a 2-panel plot with NaN filtering and annotation boxes."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        i0_values = np.array(i0_values)
        rg_values = np.array(rg_values)
        mw_values = np.array(mw_values)
        file_numbers = np.array(file_numbers) #convert to numpy array

        valid_indices_rg = ~np.isnan(i0_values) & ~np.isnan(rg_values)
        valid_indices_mw = ~np.isnan(i0_values) & ~np.isnan(mw_values)

        i0_rg_filtered = i0_values[valid_indices_rg]
        rg_filtered = rg_values[valid_indices_rg]
        i0_mw_filtered = i0_values[valid_indices_mw]
        mw_filtered = mw_values[valid_indices_mw]
        file_numbers_rg = file_numbers[valid_indices_rg]
        file_numbers_mw = file_numbers[valid_indices_mw]

        # First panel: I(0) and Rg vs. File Number
                #couleur
        colb='#0D92F4' #blue
        colr='#C62E2E' #red
        ax1 = axes[0]
        ax1.plot(file_numbers_rg, i0_rg_filtered, color=colr, label="I(0)")
        ax1.set_xlabel("File Number")
        ax1.set_ylabel("I(0)", color=colr)
        ax1.tick_params(axis="y", labelcolor=colr)

        ax1b = ax1.twinx()
        ax1b.plot(file_numbers_rg, rg_filtered, color=colb, label="Rg")
        ax1b.set_ylabel("Rg (Å)", color=colb)
        ax1b.tick_params(axis="y", labelcolor=colb)

        ax1.set_title("Ragtime : I(0) and Rg vs File Number")
        ax1.legend([f"I(0)"],loc="upper left")
        ax1b.legend([f"Rg)"],loc="upper right")
        ax1.grid(True)

        max_i0_rg = np.max(i0_rg_filtered)
        max_i0_rg_index = np.argmax(i0_rg_filtered) #get the index of the max i0
        frame_number_rg = file_numbers_rg[max_i0_rg_index+1]
        #print("frame_number_rg=",frame_number_rg)
        max_rg = rg_filtered[max_i0_rg_index] #use the index of the max i0 to get the correct Rg
        
        annotation_text_rg = (f"Frame: {int(frame_number_rg)}\n"
                            f"I(0) max: {max_i0_rg:.2f}\n"
                            f"Rg: {max_rg:.2f}\n"
                            f"qmin_offset: {qmin_idx:.0f}\n"
                            f"qmax_offset: {qmax_idx:.0f}"
                            )

        ax1.annotate(annotation_text_rg,
                    xy=(0.85, 0.72),
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="wheat", alpha=0.5),
                    ha='left')
          
            # Save data for the first panel to a text file
        file_path_rg = os.path.join(self.directory_path, "Ragtime_I0_Rg_01.txt") #use stored directory path
        with open(file_path_rg, "w") as f:
            f.write(f"qmin_offset: {qmin_idx}\tqmax_offset: {qmax_idx}\n")  # Added qmin/qmax
            f.write("File Number\tI(0)\tRg\n")
            for file_num, i0, rg in zip(file_numbers_rg, i0_rg_filtered, rg_filtered):
                f.write(f"{file_num}\t{i0:.4f}\t{rg:.4f}\n")

        # Second panel: I(0) and Mw vs. File Number
        ax2 = axes[1]
        ax2.plot(file_numbers_mw, i0_mw_filtered, color=colr, label="I(0)")
        ax2.set_xlabel("File Number")
        ax2.set_ylabel("I(0)", color=colr)
        ax2.tick_params(axis="y", labelcolor=colr)
        ax2.grid(True)
        ax2b = ax2.twinx()
        ax2b.plot(file_numbers_mw, mw_filtered, color=colb, label="Mw")
        ax2b.set_ylabel("Mw (kDa)", color=colb)
        ax2b.tick_params(axis="y", labelcolor=colb)

        ax2.set_title("Ragtime: I(0) and Mw vs File Number")
        ax2.legend([f"I(0)"],loc="upper left")
        ax2b.legend([f"MW"],loc="upper right")

        max_i0_mw = np.max(i0_mw_filtered)
        max_i0_mw_index = np.argmax(i0_mw_filtered) #get the index of the max i0
        frame_number_mw = file_numbers_mw[max_i0_mw_index + 1]
        max_mw = mw_filtered[max_i0_mw_index] #use the index of the max i0 to get the correct MW

        annotation_text_mw = (f"Frame: {int(frame_number_mw)}\n"
                            f"I(0) max: {max_i0_mw:.2f}\n"
                            f"MW: {max_mw:.2f}")

        ax2.annotate(annotation_text_mw,
                    xy=(0.87, 0.79),
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="wheat", alpha=0.5),
                    ha='left')
        # Save data for the second panel to a text file
        file_path_mw = os.path.join(self.directory_path, "Ragtime_I0_Mw_02.txt") #use stored directory path
        with open(file_path_mw, "w") as f:
            f.write(f"qmin_offset: {qmin_idx}\tqmax_offset: {qmax_idx}\n")  # Added qmin/qmax
            f.write("File Number\tI(0)\tMW\n")
            for file_num, i0, mw in zip(file_numbers_mw, i0_mw_filtered, mw_filtered):
                f.write(f"{file_num}\t{i0:.4f}\t{mw:.4f}\n")
       
        # make the plot neat 
        plt.tight_layout()
  
        # Save plots
        plot_path_png = os.path.join(self.directory_path, "Ragtime-Sec-SAXS.png")
        plot_path_svg = os.path.join(self.directory_path, "Ragtime-Sec-SAXS.svg")
        plt.savefig(plot_path_png)
        plt.savefig(plot_path_svg, format="svg")
        
        #show plot
        plt.show()


#################################
#
# Reset
#
################################# 

    def reset(self):
        """Reset the application to its initial state."""
        self.files_data = []
        self.best_file = None
        self.data = None
        self.MW = None

        self.qmin_entry.delete(0, tk.END)
        self.qmax_entry.delete(0, tk.END)

        self.folder_label.config(text="Selected Folder: None", fg="blue")
        self.file_label.config(text="No file selected", fg="black")
        self.rg_label.config(text="Rg: ")
        self.i0_label.config(text="I(0): ")
        self.qmin_rg.config(text="qmin*Rg: ")
        self.qmax_rg.config(text="qmax*Rg: ")
        self.mw_label.config(text="MW: ")

#################################
#
# Quit
#
################################# 

    """Quit action"""
    def quit(self):
        # Quit button
            root.quit()



# Main code to run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = SAXSAnalysisApp(root)
    root.mainloop()