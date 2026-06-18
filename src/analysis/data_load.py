from pathlib import Path
import pandas as pd

# 1. Point directly to your Excel file using Path
excel_file = Path("results") / "results.xlsx"

# Load the single Excel file structure
xl = pd.ExcelFile(excel_file)
sheet_names = xl.sheet_names

# Create a dictionary to hold each sheet's DataFrame for easy access
dfs = {}

# 2. Automatically loop through each sheet inside your file
for sheet in sheet_names:
    if "int_" in sheet:
        # For sheets with multi-row headers (run and hit/miss)
        df = pd.read_excel(xl, sheet_name=sheet, header=[0, 1], index_col=0)
        
        # Clean up the multi-level headers so they are perfectly aligned
        cleaned_columns = []
        current_run = ""
        for col in df.columns:
            run_level = col[0]
            status_level = col[1]
            if "Unnamed:" not in str(run_level):
                current_run = run_level
            cleaned_columns.append((current_run, status_level))
            
        df.columns = pd.MultiIndex.from_tuples(cleaned_columns, names=['run', 'status'])
    else:
        # For standard classification sheets (single header row)
        df = pd.read_excel(xl, sheet_name=sheet, index_col=0)
        df.index.name = 'subject'
    
    # Save the dataframe into our dictionary using the sheet name as the key
    dfs[sheet] = df


print(dfs)
# =========================================================================
# 3. PASTE / UPDATE YOUR NUMBERS HERE
# =========================================================================

# # Example A: Updating a classification sheet (e.g., 'no_rot_with_ref')
# dfs['no_rot_with_ref'].loc['GH 043', 'run2'] = 'hit'

# # Example B: Updating an interval sheet (e.g., 'int_no_rotation_no_ref')
# dfs['int_no_rotation_no_ref'].loc['GH 043', ('run1', 'hit')] = 12
# dfs['int_no_rotation_no_ref'].loc['GH 043', ('run1', 'miss')] = 2

