This folder contains code to convert simulated (Ex,Eg) matrices from ROOT files output by the GEANT4 simulation package (https://geant4.web.cern.ch) to Numpy arrays. It is not used within our provided example, which utilizes data already stored in Numpy arrays, but is important when initially simulating data. 

HistToCSV.C is a script to be run where the simulated ROOT files are stored. It converts bin contents for a square, 2D (Ex,Eg) matrix to a scaffolded, 1D CSV file. 

processing.ipynb contains code that unscaffolds the 1D CSV files and reconstructs the 2D matrices as Numpy arrays, which can then be easily saved and used for machine learning applications. 