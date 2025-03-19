### Ragtime-SEC-SAXS-Analysis

This script allows you to analyze and plot the elution profile of your SEC-SAXS experiment using already subtracted (.dat) files.
It creates two text files:

1- Ragtime_I0_Rg_01.txt containing three columns: Index of the frame, I(0) and Rg values,

2- Ragtime_I0_MW_02.txt containing three columns: Index of the frame, I(0) and MW values.

Then you can easily retrieve your region under your peak of interest.
You can use YOUR preferred software like Excel, Prism, SigmaPlot, etc., to plot the data as YOU want. 
If you want you can also keep the graphs displayed by the script as a picture (see Output).

Stand-alone versions can be found at the end of the Readme

Your feedback is essential to help me improve and continue this project. 
You can reach out to me directly at [reach out to me via email](jean-marie.bourhis@univ-grenoble-alpes.fr).

### User manual
# Prerequisites
   - Python 3.x
   - Necessary packages: numpy, matplotlib, scipy
     
If needed just type in your terminal :
conda install numpy matplotlib scipy 
        
   - SAXS data need to be in Å-1 (for accurate MW estimation).
   - All substracted files need to be in the same folder. 


## Features
 1. Guinier approximation :
 - Read each .dat files and determine the first usable line based on the provided qmin_offset qmax_offset.
 - Data extraction for q and I(q) in the selected range.
 - Perform a linear regression to calculate Rg (radius of gyration) and I(0) (intensity at q=0).
 - Write data to text file.
 - Display graph with I(0) and Rg vs Frame index

 2. Volume of correlation (VC):
 - Extract data up to q=0.3.
 - Calculate the integral of the product I(q)*q.
 - Calculation of VC, QR (quality factor) and MW (molecular weight).
 - Write data to a text file.
 - Display graph with I(0) and MW vs Frame index


## Command syntax
To launch the script in your terminal type:

 python Ragtime-v2.0.py

A gui will appear 

![image](https://github.com/user-attachments/assets/e15e642a-2e23-41b8-8d5d-e5abfbde30b6)


   - Browse :  to find the location of the folder containing ONLY the Substracted files from SEC-SAXS.
   - `qmin_offset`: the offset (in a number of lines) to determine qmin, (like for PRIMUS "range or RAW "nmin").
   - `qmax_offset`: the offset (in a number of lines) to determine qmax, (like for PRIMUS "range or RAW "nmax").
   - Auto-Guinier: will try to find a suitable guinier zone on the file with the highest intensity (not as good as Primus or Raw but ..).
   - Manual-Guinier: If you know q_offsets then fill the value and check the result
   - Ragtime: will process all the Sub.dat files to calculate Rg, I(0) and Mw
   - Reset : reset everything to the strat
   - Quit : allow you to close the gui
     
## Browse

   - it will point to your folder with all your already substracted files
   - Check headers of the files
   - look for the file call best files with the highest intensity at q=0.1 (arbitrary choice)
     
## Auto Guinier 

   - will  find the guinier region
   - displays 4 plots:  1- the Form factor
                       2- the Guinier region with its residuals
                       3- normalized krtky plot
                       4- Volume of correlation to estimate the MW

![image](https://github.com/user-attachments/assets/852d89e8-67fe-47fd-b32e-0e990912a0e7)

     
## Manual Guinier 
   - will do the same as Auto-Guinier but you need to fill the qmin_offsets) before using it

## Ragtime

- Process alls the files
- Diplays a 2 plots : 1- I(0) and Rg vs frame number
                      2- I(0) and MW vs frame number

![image](https://github.com/user-attachments/assets/7381ac6b-af85-4253-9f3e-8ada251f717f)

 
## Tests
It has been tested only on SEC-SAXS data from SWING beamline at Soleil-synchrotron France.
(but should work with files from other SEC-SAXS beamlines)

## Stand alone versions :

Windows is here :
coming soon

MAc.app is here:  (if it doesn't start go to Privacy & Security and click on open anyway) 
coming soon

Linux is here:
coming soon 


Again your feedback is essential to help me improve and continue this project. 
You can reach out to me directly at [reach out to me via email](jean-marie.bourhis@univ-grenoble-alpes.fr).


# NB: I named the script Ragtime as a nod to Foxtrot usesd on SWING beamline at Soleil-synchrotron France, the software that inspired it. Just as the Foxtrot dance was originally performed to ragtime music. Ragtime follows the same philosophy as Foxtrot for processing Sec-SAXS data .
