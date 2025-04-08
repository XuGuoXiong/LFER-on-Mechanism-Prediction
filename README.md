## README

This folder contains the code and the bond formation mechanism prediction results for 132,300 combinations involved in the article "Unveiling Mechanistic Patterns of Radical Bond Formation in Copper-Catalyzed Radical Transformation Reactions through Predictive Linear Free Energy Relationship Modelling". The folder contains the following files: 

## Dependences
The code in this folder requires the following Python libraries:
- python = 3.11.5
- numpy == 1.24.3
- pandas == 2.2.2
- sklearn >= 1.5.2

## Usage
1. Solve_Equation_Groups.py: Python code for solving systems of equations using the least squares method, which is used to fit the reactivity parameters for each reaction component from the free energy barriers obtainable in the initial set (for details, see the article).
- Input file examples: dG_RE_num.xlsx, dG_RS_num.xlsx, dG_IP_num.xlsx (in LFER-on-Mechanism-Prediction/Free_Energy_Changes_in_Different_Bond_Formation_Progresses/)

Before using this script, you need to provide a formatted data file (in xlsx format) of free energy barriers within the initial set: the first column should contain the name or serial number of the ligand, the second column should contain the name or serial number of the radical, the third column should contain the name or serial number of the nucleophile, and the fourth column should contain the values of the free energy barrier corresponding to a specific bond formation mechanism for that ligand-radical-nucleophile combination. Note that a single list should only contain free energy data for one type of bond formation mechanism; and if the free energy barrier for a certain combination is not available (for example, because its transition state is hard to be located), the value for that entry should be NaN (not "-" or "None", etc.).

2. Multiple_Linear_Regression.py: Python code for multiple linear regression, used to establish a fit between reactivity parameters and physico-organic parameters.
- Input file examples: R_vs_Parameters.xlsx, N_vs_Parameters.xlsx (in LFER-on-Mechanism-Prediction/MLR_Parameters/)

Before using this script, you need to provide a formatted data file (in CSV format) that includes reactivity parameters and other physical organic parameters: the first column should contain the reactivity parameter for a specific reaction component under a certain mechanism (for example, the reactivity parameter R_RE for radicals under reductive elimination), and from the second column onwards should contain the values of the physical organic parameters corresponding to that reaction component. Note that the name or serial number of the reaction component does not need to be provided; it is only necessary to align the reactivity parameter and physical organic parameters of the same reaction component in the same row; all data must not be empty.

3. Distinguished_Mechanisms_Sorted_SMILES-version.xlsx: Prediction results for the bond formation mechanisms of 132,300 ligand-radical-nucleophile combinations, predicted based on the linear free energy relationship predictive model and the calibrated reactivity parameters established in the article.

In this file, the serial numbers and SMILES of ligands, radicals, and nucleophiles are provided, which can assist users in identifying the bond formation mechanism selectivity for specific reactant combinations. If the free energy barrier of the most preferred mechanism is more than 3 kcal/mol lower than those of other mechanisms, it will be considered dominant. On the contrary, if this is not the case, it will be seen as indicative of competition among two or more mechanisms (the names of the competing mechanisms will be noted).
