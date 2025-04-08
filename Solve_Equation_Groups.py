import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import leastsq

file = Path("file.xlsx") # dG_RE_num.xlsx, dG_RS_num.xlsx, dG_IP_num.xlsx
df = pd.read_excel(file, engine="openpyxl")

# Retrieve the labels of the independent variables
ligand_labels = df["ligand"].dropna().unique()
radical_labels = df["radical"].dropna().unique()
nucleophile_labels = df["nucleophile"].dropna().unique()

# Create the coefficient matrix and the right-hand side vector
coefficient_matrix = []
G = []

# Populate the coefficient matrix and the right-hand side vector
for _, row in df.iterrows():
    if pd.notna(row["dG"]):
        i = np.where(ligand_labels == row["ligand"])[0][0]
        j = np.where(radical_labels == row["radical"])[0][0]
        k = np.where(nucleophile_labels == row["nucleophile"])[0][0]
        coefficients = [0] * (
            len(ligand_labels) + len(radical_labels) + len(nucleophile_labels)
        )
        coefficients[i] = 1
        coefficients[len(ligand_labels) + j] = 1
        coefficients[len(ligand_labels) + len(radical_labels) + k] = 1
        coefficient_matrix.append(coefficients)
        G.append(row["dG"])

coefficient_matrix = np.array(coefficient_matrix)
G = np.array(G)
G = -G

# For the ternary linear equation G = L + R + N, there exists a scenario where constant values are assigned to the independent variables (L, R, N), allowing potential linear shifts in their absolute magnitudes.
# For instance, under transformations L' = L + 10, R' = R - 3, N' = N - 3, the equation G = L' + R' + N' still holds true. Different methodologies or initial guesses may lead to variations in these constant value assignments.
# Nevertheless, the relative values of LFER parameters within L, R, and N remain fundamentally consistent.
# Therefore, we adopt the strategy of fixing a single parameter value to obtain precise and stable absolute numerical solutions for L, R, and N.

def solve(fixed_x_val, fixed_y_val, fixed_z_val, mechanism):
    fixed_x_index = 0
    fixed_y_index = len(ligand_labels)
    fixed_z_index = len(ligand_labels) + len(radical_labels)
    fixed_indices = [fixed_x_index, fixed_y_index, fixed_z_index]
    fixed_values = [fixed_x_val, fixed_y_val, fixed_z_val]

    A_fixed = coefficient_matrix[:, fixed_indices]
    adjusted_G = G - np.dot(A_fixed, fixed_values)

    free_columns_mask = np.ones(coefficient_matrix.shape[1], dtype=bool)
    free_columns_mask[fixed_indices] = False
    A_free = coefficient_matrix[:, free_columns_mask]

    try:
        params_free, _, _, _ = np.linalg.lstsq(A_free, adjusted_G, rcond=None)
    except np.linalg.LinAlgError:
        print("The matrix cannot be solved.")
        exit()

    solution = np.zeros(coefficient_matrix.shape[1])
    solution[fixed_indices] = fixed_values
    solution[free_columns_mask] = params_free

    results = pd.DataFrame({
        "Ligand": [ligand_labels[i] for i in range(len(ligand_labels))] 
                + [radical_labels[i] for i in range(len(radical_labels))] 
                + [nucleophile_labels[i] for i in range(len(nucleophile_labels))],
        "Value": np.concatenate((
            solution[:len(ligand_labels)],
            solution[len(ligand_labels):len(ligand_labels)+len(radical_labels)],
            solution[-len(nucleophile_labels):]
        ))
    })

    results.to_csv(f"solution_output_{mechanism}.csv", index=False)
    print(f"The results have been saved to 'solution_output_{mechanism}.csv'")

if "RE" in file.name and "RS" not in file.name and "IP" not in file.name:
    mechanism = "RE"
    fixed_x_val = 83.6645139058837
    fixed_y_val = 53.9781838108333
    fixed_z_val = -17.3958610938274
    solve(fixed_x_val, fixed_y_val, fixed_z_val, mechanism)
elif "RS" in file.name and "RE" not in file.name and "IP" not in file.name:
    mechanism = "RS"
    fixed_x_val = 40.2591628371676
    fixed_y_val = 11.3134219163921
    fixed_z_val = 62.2725422247131
    solve(fixed_x_val, fixed_y_val, fixed_z_val, mechanism)
elif "IP" in file.name and "RE" not in file.name and "RS" not in file.name:
    mechanism = "IP"
    fixed_x_val = 56.6284566174279
    fixed_y_val = 73.543980361312
    fixed_z_val = 5.06948540812359
    solve(fixed_x_val, fixed_y_val, fixed_z_val, mechanism)
else:
    print("Need formatted input file.")
