import json
import nbformat
from pathlib import Path

nb_path = Path("nfl_offensive_clustering.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Clean up any unexpected keys in markdown cells
for cell in nb.cells:
    if cell.cell_type == 'markdown':
        cell.pop('outputs', None)
        cell.pop('execution_count', None)

# Validate notebook against nbformat schema
nbformat.validate(nb)

# Re-save cleaned notebook
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Validation successful! Cleaned notebook has {len(nb.cells)} cells.")

# Summary of executed code cells and outputs
code_cells = [c for c in nb.cells if c.cell_type == 'code']
outputs_count = sum(len(c.get('outputs', [])) for c in code_cells)
print(f"Total code cells: {len(code_cells)}")
print(f"Total outputs generated: {outputs_count}")
for i, c in enumerate(code_cells):
    print(f"  Code cell {i+1:02d}: {len(c.get('outputs', []))} output(s)")

