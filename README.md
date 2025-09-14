
"""
RDKit Molecular Visualization Script for NSAIDs
Created for medicinal chemistry students
This script demonstrates various ways to visualize molecules using RDKit
"""

# Import necessary libraries
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import numpy as np

# Enable better molecule rendering in Jupyter notebooks
IPythonConsole.ipython_useSVG = True

# Define SMILES strings for 10 common NSAIDs
# Each SMILES represents the molecular structure in text format
nsaid_smiles = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 
    "Naproxen": "COC1=CC2=C(C=C1)C=C(C=C2)C(C)C(=O)O",
    "Diclofenac": "O=C(O)CC1=CC=CC=C1NC2=C(Cl)C=CC=C2Cl",
    "Indomethacin": "COC1=CC2=C(C=C1)C(=CN2C(=O)C3=CC=C(C=C3)Cl)C(=O)O",
    "Celecoxib": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
    "Meloxicam": "CN1C2=C(C=C(C=C2)S(=O)(=O)N)N=C1C(=O)NC3=CC=CC=C3",
    "Piroxicam": "CN1C(=O)C(=CN=C1C2=CC=CC=N2)C(=O)NC3=CC=C(C=C3)S(=O)(=O)N",
    "Ketoprofen": "CC(C(=O)O)C1=CC=CC(=C1)C(=O)C2=CC=CC=C2",
    "Flurbiprofen": "CC(C(=O)O)C1=CC=C(C=C1)C2=CC=CC=C2F"
}

print("NSAIDs Molecular Visualization with RDKit")
print("=" * 50)

# Step 1: Convert SMILES to RDKit molecule objects
# This creates internal RDKit representation of molecules
molecules = {}
for name, smiles in nsaid_smiles.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:  # Check if SMILES parsing was successful
        molecules[name] = mol
        print(f"✓ Successfully parsed {name}")
    else:
        print(f"✗ Failed to parse {name}")

print(f"\nTotal molecules successfully parsed: {len(molecules)}")

# Step 2: Calculate basic molecular properties
# These properties are important in drug discovery
print("\nBasic Molecular Properties:")
print("-" * 40)
print(f"{'Name':<15} {'MW':<8} {'LogP':<8} {'HBD':<6} {'HBA':<6}")
print("-" * 40)

for name, mol in molecules.items():
    mw = Descriptors.MolWt(mol)          # Molecular weight
    logp = Descriptors.MolLogP(mol)      # Lipophilicity (LogP)
    hbd = Descriptors.NumHDonors(mol)    # Hydrogen bond donors
    hba = Descriptors.NumHAcceptors(mol) # Hydrogen bond acceptors
    print(f"{name:<15} {mw:<8.1f} {logp:<8.2f} {hbd:<6} {hba:<6}")

# Step 3: Individual molecule visualization
# This creates separate images for each molecule
print("\n" + "=" * 50)
print("INDIVIDUAL MOLECULE VISUALIZATION")
print("=" * 50)

def visualize_single_molecule(mol, name, size=(300, 300)):
    """
    Visualize a single molecule with its name

    Parameters:
    mol: RDKit molecule object
    name: String name of the molecule  
    size: Tuple of (width, height) for image size

    Returns:
    PIL Image object
    """
    img = Draw.MolToImage(mol, size=size)
    return img

# Generate individual molecule images
individual_images = {}
for name, mol in molecules.items():
    img = visualize_single_molecule(mol, name)
    individual_images[name] = img
    print(f"Generated image for {name}")

# Step 4: Grid visualization of all molecules
# This creates a single image showing all molecules together
print("\n" + "=" * 50) 
print("GRID VISUALIZATION")
print("=" * 50)

def create_molecule_grid(molecules_dict, mols_per_row=5, mol_size=(200, 200)):
    """
    Create a grid layout showing multiple molecules

    Parameters:
    molecules_dict: Dictionary of {name: mol_object}
    mols_per_row: Number of molecules per row
    mol_size: Size of each individual molecule image

    Returns:
    PIL Image object containing the grid
    """
    mol_list = list(molecules_dict.values())
    legends = list(molecules_dict.keys())

    # Create grid image with legends
    img = Draw.MolsToGridImage(
        mol_list,
        molsPerRow=mols_per_row,
        subImgSize=mol_size,
        legends=legends,
        useSVG=False  # Set to True for better quality in Jupyter notebooks
    )
    return img

# Create grid visualization
grid_img = create_molecule_grid(molecules, mols_per_row=3, mol_size=(250, 250))
print("Grid visualization created successfully!")

# Step 5: Highlighted substructure visualization
# This highlights specific functional groups or substructures
print("\n" + "=" * 50)
print("SUBSTRUCTURE HIGHLIGHTING") 
print("=" * 50)

def highlight_substructure(mol, pattern_smarts, name):
    """
    Highlight a specific substructure within a molecule

    Parameters:
    mol: RDKit molecule object
    pattern_smarts: SMARTS pattern to search for
    name: Name for display

    Returns:
    PIL Image with highlighted substructure
    """
    pattern = Chem.MolFromSmarts(pattern_smarts)
    if pattern:
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            # Highlight the first match found
            img = Draw.MolToImage(mol, highlightAtoms=matches[0], size=(300, 300))
            print(f"Found and highlighted pattern in {name}")
            return img
    print(f"Pattern not found in {name}")
    return None

# Example: Highlight carboxylic acid groups (common in NSAIDs)
carboxylic_acid_pattern = "C(=O)O"  # SMARTS pattern for carboxylic acid
print(f"Highlighting carboxylic acid groups using pattern: {carboxylic_acid_pattern}")

highlighted_images = {}
for name, mol in molecules.items():
    highlighted_img = highlight_substructure(mol, carboxylic_acid_pattern, name)
    if highlighted_img:
        highlighted_images[name] = highlighted_img

print(f"Successfully highlighted {len(highlighted_images)} molecules")

# Step 6: Save images to files (optional)
print("\n" + "=" * 50)
print("SAVING IMAGES")
print("=" * 50)

def save_molecule_images(images_dict, prefix="nsaid"):
    """
    Save molecule images to PNG files

    Parameters:
    images_dict: Dictionary of {name: PIL_image}
    prefix: Prefix for filename
    """
    saved_count = 0
    for name, img in images_dict.items():
        try:
            filename = f"{prefix}_{name.lower().replace(' ', '_')}.png"
            img.save(filename)
            print(f"Saved: {filename}")
            saved_count += 1
        except Exception as e:
            print(f"Error saving {name}: {e}")
    return saved_count

# Save individual molecule images
print("Saving individual molecule images:")
saved_individual = save_molecule_images(individual_images, "nsaid_individual")

# Save grid image
try:
    grid_img.save("nsaid_grid_visualization.png")
    print("Saved: nsaid_grid_visualization.png")
    grid_saved = True
except Exception as e:
    print(f"Error saving grid image: {e}")
    grid_saved = False

# Step 7: Advanced visualization with custom properties
print("\n" + "=" * 50)
print("ADVANCED VISUALIZATION WITH PROPERTIES")
print("=" * 50)

def create_property_colored_grid(molecules_dict, property_func, color_map='viridis'):
    """
    Create a grid where molecules are colored based on a property

    Parameters:
    molecules_dict: Dictionary of molecules
    property_func: Function to calculate property (e.g., Descriptors.MolWt)
    color_map: Matplotlib colormap name
    """
    mol_list = list(molecules_dict.values())
    legends = list(molecules_dict.keys())

    # Calculate properties for color mapping
    properties = [property_func(mol) for mol in mol_list]

    print(f"Property range: {min(properties):.2f} - {max(properties):.2f}")

    # Create the grid (RDKit doesn't directly support property-based coloring)
    # This would require more advanced customization
    img = Draw.MolsToGridImage(
        mol_list,
        molsPerRow=3,
        subImgSize=(200, 200),
        legends=[f"{name}\n{prop:.1f}" for name, prop in zip(legends, properties)]
    )
    return img

# Create molecular weight-based visualization
mw_grid = create_property_colored_grid(molecules, Descriptors.MolWt)
print("Created molecular weight-based grid visualization")

# Summary statistics
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Total NSAIDs processed: {len(molecules)}")
print(f"Individual images created: {len(individual_images)}")
print(f"Highlighted images created: {len(highlighted_images)}")
print("Grid visualizations: 2 (standard and property-based)")
print(f"Images saved to disk: {saved_individual + (1 if grid_saved else 0)}")

print("\n" + "=" * 50)
print("USAGE NOTES")
print("=" * 50)
print("1. Install RDKit: conda install -c conda-forge rdkit")
print("2. For Jupyter notebooks, the images will display inline")
print("3. Modify mol_size and mols_per_row parameters as needed")
print("4. Use different SMARTS patterns for highlighting different substructures")
print("5. Properties can be calculated using Descriptors module")
print("6. Images are saved as PNG files in the current directory")

print("\nScript completed successfully! 🎉")
