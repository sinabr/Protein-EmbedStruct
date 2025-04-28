import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import PDB
from Bio.PDB import PDBParser, Superimposer
from scipy.spatial.distance import pdist, squareform, cosine
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from transformers import T5EncoderModel, T5Tokenizer
import esm
import os
import subprocess
import requests
from io import StringIO

# Set up global variables
PDB_DIR = "pdb_files/"
RESULTS_DIR = "results/"
os.makedirs(PDB_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========================== PART 1: Data Acquisition ==========================

def download_pdb(pdb_id, output_dir=PDB_DIR):
    """Download a PDB file from the RCSB PDB database."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    out_file = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
    
    if os.path.exists(out_file):
        print(f"PDB file {pdb_id} already exists.")
        return out_file
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(out_file, 'w') as f:
            f.write(response.text)
        print(f"Downloaded {pdb_id} successfully")
        return out_file
    else:
        print(f"Failed to download {pdb_id}, status code: {response.status_code}")
        return None

def extract_sequence_from_pdb(pdb_file):
    """Extract protein sequence from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    # Get the first model
    model = structure[0]
    
    # Extract sequence
    sequence = ""
    for chain in model:
        for residue in chain:
            if residue.get_id()[0] == ' ':  # Skip hetero-atoms and water
                resname = residue.get_resname()
                one_letter = PDB.Polypeptide.three_to_one(resname) if resname in PDB.Polypeptide.standard_aa_names else 'X'
                sequence += one_letter
    
    return sequence

def get_protein_pairs():
    """Define pairs of proteins for comparative analysis."""
    # Example protein pairs with wild type and mutant/variant
    protein_pairs = [
        # Lysozyme variants
        ("2lzm", "1l63"),  # T157I mutant
        # GFP variants
        ("1ema", "1emb"),  # Wild-type GFP and S65T variant
        # Beta-lactamase variants
        ("1btl", "1fqg"),  # Wild-type TEM-1 beta-lactamase and a variant
        # HIV protease variants
        ("3hvp", "4hvp"),  # HIV-1 protease variants
        # Hemoglobin variants
        ("1hho", "1hbr"),  # Hemoglobin variants
        # NRAS variants
        ("3con", "4g0n"),  # NRAS wild-type and mutant
    ]
    
    return protein_pairs

def get_multi_domain_proteins():
    """Define multi-domain proteins for analysis."""
    # List of PDB IDs for proteins with well-defined multiple domains
    multi_domain = ["1avn", "1cll", "2bbm", "3pdk"]  # Examples: kinases, calcium-binding proteins, etc.
    
    # List of PDB IDs for proteins with single domains for comparison
    single_domain = ["1ubq", "1lmb", "3hhr", "1fas"]  # Examples: ubiquitin, lambda repressor, etc.
    
    return multi_domain, single_domain

# ========================== PART 2: RMSD Calculation ==========================

def calculate_rmsd(pdb_id1, pdb_id2, chain1='A', chain2='A'):
    """
    Calculate RMSD between two protein structures using BioPython.
    Returns RMSD value and aligned atoms.
    """
    # Download PDB files if not already present
    pdb_file1 = download_pdb(pdb_id1)
    pdb_file2 = download_pdb(pdb_id2)
    
    if not pdb_file1 or not pdb_file2:
        return None, None
    
    # Parse structures
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure(pdb_id1, pdb_file1)
    structure2 = parser.get_structure(pdb_id2, pdb_file2)
    
    # Get atoms for alignment (CA atoms only)
    atoms1 = []
    atoms2 = []
    
    # Only consider residues present in both structures
    res_dict1 = {res.get_id()[1]: res for res in structure1[0][chain1] if res.get_id()[0] == ' '}
    res_dict2 = {res.get_id()[1]: res for res in structure2[0][chain2] if res.get_id()[0] == ' '}
    
    # Find common residues
    common_res = set(res_dict1.keys()).intersection(set(res_dict2.keys()))
    
    for res_id in sorted(common_res):
        if 'CA' in res_dict1[res_id] and 'CA' in res_dict2[res_id]:
            atoms1.append(res_dict1[res_id]['CA'])
            atoms2.append(res_dict2[res_id]['CA'])
    
    if len(atoms1) < 3:  # Need at least 3 pairs for superposition
        print(f"Not enough matching CA atoms between {pdb_id1} and {pdb_id2}")
        return None, None
    
    # Perform superposition
    super_imposer = Superimposer()
    super_imposer.set_atoms(atoms1, atoms2)
    super_imposer.apply(structure2[0][chain2].get_atoms())
    
    return super_imposer.rms, len(atoms1)

def calculate_rmsd_with_pymol(pdb_id1, pdb_id2):
    """
    Alternative method: Calculate RMSD using PyMOL (requires PyMOL installed).
    Note: This is a shell-out to PyMOL, which must be installed and accessible.
    """
    # Download PDB files if needed
    pdb_file1 = download_pdb(pdb_id1)
    pdb_file2 = download_pdb(pdb_id2)
    
    # PyMOL script to calculate RMSD
    pymol_script = f"""
    import pymol
    from pymol import cmd
    
    # Load structures
    cmd.load("{pdb_file1}", "prot1")
    cmd.load("{pdb_file2}", "prot2")
    
    # Align structures
    rmsd = cmd.align("prot1 and name CA", "prot2 and name CA")[0]
    
    # Print RMSD
    print(f"RMSD: {{rmsd}}")
    
    # Exit PyMOL
    cmd.quit()
    """
    
    # Write the PyMOL script to a temporary file
    with open("temp_pymol_script.py", "w") as f:
        f.write(pymol_script)
    
    # Execute PyMOL with the script
    try:
        result = subprocess.run(["pymol", "-cq", "temp_pymol_script.py"], 
                                capture_output=True, text=True)
        
        # Parse the output to extract RMSD
        for line in result.stdout.split('\n'):
            if "RMSD:" in line:
                rmsd = float(line.split(":")[1].strip())
                return rmsd
        
        print("Failed to extract RMSD from PyMOL output")
        return None
    except Exception as e:
        print(f"Error running PyMOL: {e}")
        return None
    finally:
        # Clean up
        if os.path.exists("temp_pymol_script.py"):
            os.remove("temp_pymol_script.py")

# ========================== PART 3: Embedding Models ==========================

def load_esm2_model():
    """Load the ESM-2 model for protein embeddings."""
    # Load ESM-2 (15B version if available, otherwise use smaller)
    try:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # Set to evaluation mode
        return model, batch_converter
    except:
        print("Failed to load ESM-2, please install it with: pip install fair-esm")
        return None, None

def load_prot_t5_model():
    """Load the ProtT5 model for protein embeddings."""
    try:
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        model.eval()
        return model, tokenizer
    except:
        print("Failed to load ProtT5, please install transformers and related packages")
        return None, None

def get_esm2_embedding(sequence, model, batch_converter):
    """Get ESM-2 embedding for a protein sequence."""
    if model is None or batch_converter is None:
        return None
    
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    
    # Extract per-residue embeddings (last layer)
    token_representations = results["representations"][33]
    
    # Mean-pool over residues to get a fixed-size embedding
    sequence_embedding = token_representations[0, 1:len(sequence)+1].mean(dim=0)
    
    return sequence_embedding.numpy()

def get_prot_t5_embedding(sequence, model, tokenizer):
    """Get ProtT5 embedding for a protein sequence."""
    if model is None or tokenizer is None:
        return None
    
    # Add spaces between amino acids for tokenization
    sequence = " ".join(list(sequence))
    
    # Tokenize
    token_encoding = tokenizer.batch_encode_plus([sequence], 
                                                add_special_tokens=True, 
                                                padding="longest", 
                                                return_tensors="pt")
    
    # Generate embedding
    with torch.no_grad():
        embedding = model(input_ids=token_encoding['input_ids'], 
                          attention_mask=token_encoding['attention_mask'])
    
    # Mean-pool over residues to get a fixed-size embedding
    embedding = embedding.last_hidden_state.mean(dim=1)
    
    return embedding.numpy()[0]

def generate_embeddings_for_proteins(pdb_ids):
    """Generate embeddings for a list of proteins using multiple models."""
    # Load models
    esm2_model, esm2_converter = load_esm2_model()
    prot_t5_model, prot_t5_tokenizer = load_prot_t5_model()
    
    results = {}
    
    for pdb_id in pdb_ids:
        # Download and extract sequence
        pdb_file = download_pdb(pdb_id)
        if not pdb_file:
            continue
        
        sequence = extract_sequence_from_pdb(pdb_file)
        
        # Generate embeddings
        esm2_emb = get_esm2_embedding(sequence, esm2_model, esm2_converter)
        prot_t5_emb = get_prot_t5_embedding(sequence, prot_t5_model, prot_t5_tokenizer)
        
        results[pdb_id] = {
            'sequence': sequence,
            'esm2_embedding': esm2_emb,
            'prot_t5_embedding': prot_t5_emb
        }
    
    return results

# ========================== PART 4: Analysis Functions ==========================

def calculate_embedding_distances(embeddings_dict, method='cosine'):
    """
    Calculate distances between protein embeddings.
    
    Args:
        embeddings_dict: Dictionary with protein IDs as keys and embeddings as values
        method: Distance metric ('cosine', 'euclidean', etc.)
    
    Returns:
        Dictionary of distance matrices for each embedding type
    """
    pdb_ids = list(embeddings_dict.keys())
    
    # Initialize results dictionary
    distance_matrices = {}
    
    # Get available embedding types from the first item
    embedding_types = [k for k in embeddings_dict[pdb_ids[0]].keys() if 'embedding' in k]
    
    for emb_type in embedding_types:
        # Extract embeddings in the same order as pdb_ids
        embeddings = [embeddings_dict[pdb_id][emb_type] for pdb_id in pdb_ids]
        
        # Skip if embeddings are not available
        if any(emb is None for emb in embeddings):
            print(f"Skipping {emb_type} due to missing embeddings")
            continue
        
        # Calculate pairwise distances
        if method == 'cosine':
            # For cosine, we need 1 - cosine similarity for distance
            distances = np.zeros((len(pdb_ids), len(pdb_ids)))
            for i in range(len(pdb_ids)):
                for j in range(len(pdb_ids)):
                    distances[i, j] = cosine(embeddings[i], embeddings[j])
        else:
            distances = squareform(pdist(embeddings, metric=method))
        
        # Create DataFrame with protein IDs as labels
        df = pd.DataFrame(distances, index=pdb_ids, columns=pdb_ids)
        distance_matrices[emb_type] = df
    
    return distance_matrices

def compare_rmsd_vs_embedding_distance(protein_pairs, embedding_results):
    """
    Compare RMSD values with embedding distances for protein pairs.
    
    Args:
        protein_pairs: List of PDB ID pairs (tuples)
        embedding_results: Dictionary with embedding results
        
    Returns:
        DataFrame with RMSD and embedding distances
    """
    results = []
    
    for pdb_id1, pdb_id2 in protein_pairs:
        # Calculate RMSD
        rmsd, n_atoms = calculate_rmsd(pdb_id1, pdb_id2)
        
        if rmsd is None:
            print(f"Skipping pair {pdb_id1}-{pdb_id2} due to RMSD calculation failure")
            continue
        
        # Check if embeddings are available
        if pdb_id1 not in embedding_results or pdb_id2 not in embedding_results:
            print(f"Skipping pair {pdb_id1}-{pdb_id2} due to missing embeddings")
            continue
        
        # Calculate embedding distances
        emb_distances = {}
        for emb_type in [k for k in embedding_results[pdb_id1].keys() if 'embedding' in k]:
            emb1 = embedding_results[pdb_id1][emb_type]
            emb2 = embedding_results[pdb_id2][emb_type]
            
            if emb1 is None or emb2 is None:
                emb_distances[emb_type] = None
            else:
                emb_distances[emb_type] = cosine(emb1, emb2)
        
        # Add result
        result = {
            'pdb_id1': pdb_id1,
            'pdb_id2': pdb_id2,
            'rmsd': rmsd,
            'n_aligned_atoms': n_atoms
        }
        result.update(emb_distances)
        
        results.append(result)
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def analyze_mutation_impact(mutant_pairs, embedding_results):
    """
    Analyze the impact of mutations on protein embeddings.
    
    Args:
        mutant_pairs: List of pairs (wild_type_pdb, mutant_pdb, mutation_info)
        embedding_results: Dictionary with embedding results
    
    Returns:
        DataFrame with mutation impact on embeddings
    """
    results = []
    
    for wt_pdb, mut_pdb in mutant_pairs:
        # Skip if embeddings are not available
        if wt_pdb not in embedding_results or mut_pdb not in embedding_results:
            print(f"Skipping {wt_pdb}-{mut_pdb} due to missing embeddings")
            continue
        
        # Calculate embedding differences
        for emb_type in [k for k in embedding_results[wt_pdb].keys() if 'embedding' in k]:
            wt_emb = embedding_results[wt_pdb][emb_type]
            mut_emb = embedding_results[mut_pdb][emb_type]
            
            if wt_emb is None or mut_emb is None:
                continue
                
            # Calculate cosine distance
            cos_dist = cosine(wt_emb, mut_emb)
            
            # Calculate Euclidean distance
            euc_dist = np.linalg.norm(wt_emb - mut_emb)
            
            results.append({
                'wild_type': wt_pdb,
                'mutant': mut_pdb,
                'embedding_type': emb_type,
                'cosine_distance': cos_dist,
                'euclidean_distance': euc_dist
            })
    
    return pd.DataFrame(results)

def visualize_domain_separation(embedding_results, domain_annotations):
    """
    Visualize domain separation in embedding space using t-SNE.
    
    Args:
        embedding_results: Dictionary with embedding results
        domain_annotations: Dictionary mapping PDB IDs to domain information
        
    Returns:
        Dictionary with t-SNE results for each embedding type
    """
    tsne_results = {}
    
    # Get available embedding types
    pdb_id = list(embedding_results.keys())[0]
    embedding_types = [k for k in embedding_results[pdb_id].keys() if 'embedding' in k]
    
    for emb_type in embedding_types:
        # Extract PDB IDs and embeddings
        pdb_ids = []
        embeddings = []
        domain_labels = []
        
        for pdb_id, data in embedding_results.items():
            if data[emb_type] is not None and pdb_id in domain_annotations:
                pdb_ids.append(pdb_id)
                embeddings.append(data[emb_type])
                domain_labels.append(domain_annotations[pdb_id])
        
        if len(embeddings) < 3:
            print(f"Not enough valid embeddings for {emb_type}")
            continue
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        tsne_data = tsne.fit_transform(np.array(embeddings))
        
        # Create DataFrame with results
        df = pd.DataFrame({
            'pdb_id': pdb_ids,
            'domain_type': domain_labels,
            'tsne_1': tsne_data[:, 0],
            'tsne_2': tsne_data[:, 1]
        })
        
        tsne_results[emb_type] = df
    
    return tsne_results

def test_embedding_disentanglement(embedding_results, perturb_functions):
    """
    Test whether embedding dimensions correspond to specific structural features.
    
    Args:
        embedding_results: Dictionary with embedding results
        perturb_functions: List of functions that perturb sequences in specific ways
    
    Returns:
        DataFrame with perturbation response scores for embedding dimensions
    """
    results = []
    
    for pdb_id, data in embedding_results.items():
        sequence = data['sequence']
        
        # Skip if sequence is not available
        if not sequence:
            continue
        
        # Apply each perturbation function to the sequence
        for perturb_name, perturb_func in perturb_functions.items():
            perturbed_seq = perturb_func(sequence)
            
            # Generate embeddings for perturbed sequence
            perturbed_embeddings = {}
            
            # Check if ESM-2 model is available
            esm2_model, esm2_converter = load_esm2_model()
            if esm2_model is not None:
                perturbed_embeddings['esm2_embedding'] = get_esm2_embedding(perturbed_seq, esm2_model, esm2_converter)
            
            # Check if ProtT5 model is available
            prot_t5_model, prot_t5_tokenizer = load_prot_t5_model()
            if prot_t5_model is not None:
                perturbed_embeddings['prot_t5_embedding'] = get_prot_t5_embedding(perturbed_seq, prot_t5_model, prot_t5_tokenizer)
            
            # Calculate dimension-wise differences
            for emb_type in [k for k in data.keys() if 'embedding' in k]:
                if emb_type in perturbed_embeddings and perturbed_embeddings[emb_type] is not None:
                    original_emb = data[emb_type]
                    perturbed_emb = perturbed_embeddings[emb_type]
                    
                    # Calculate dimension-wise differences
                    dim_diffs = np.abs(original_emb - perturbed_emb)
                    
                    # Get top affected dimensions
                    top_dims = np.argsort(dim_diffs)[-10:]  # Top 10 most affected dimensions
                    
                    for dim in top_dims:
                        results.append({
                            'pdb_id': pdb_id,
                            'embedding_type': emb_type,
                            'perturbation': perturb_name,
                            'dimension': dim,
                            'difference': dim_diffs[dim]
                        })
    
    return pd.DataFrame(results)

# ========================== PART 5: Perturbation Functions ==========================

def perturb_n_terminal(sequence, length=5):
    """Perturb N-terminal region of protein."""
    if len(sequence) <= length:
        return sequence
    
    # Replace N-terminal residues with alanines
    return "A" * length + sequence[length:]

def perturb_c_terminal(sequence, length=5):
    """Perturb C-terminal region of protein."""
    if len(sequence) <= length:
        return sequence
    
    # Replace C-terminal residues with alanines
    return sequence[:-length] + "A" * length

def perturb_hydrophobic_core(sequence):
    """
    Perturb hydrophobic residues in the sequence.
    This is a simplified version - in reality you'd want to use 
    structural information to identify the core.
    """
    hydrophobic = "ACFILMVWY"
    perturbed = ""
    
    for aa in sequence:
        if aa in hydrophobic:
            perturbed += "A"  # Replace with alanine
        else:
            perturbed += aa
    
    return perturbed

def perturb_surface_residues(sequence):
    """
    Perturb likely surface residues in the sequence.
    Simplified version - assumes charged and polar residues are on surface.
    """
    surface = "DEHKNQRST"
    perturbed = ""
    
    for aa in sequence:
        if aa in surface:
            perturbed += "S"  # Replace with serine
        else:
            perturbed += aa
    
    return perturbed

def get_perturbation_functions():
    """Get dictionary of perturbation functions."""
    return {
        "n_terminal": perturb_n_terminal,
        "c_terminal": perturb_c_terminal,
        "hydrophobic_core": perturb_hydrophobic_core,
        "surface_residues": perturb_surface_residues
    }

# ========================== PART 6: Main Pipeline ==========================

def run_complete_analysis():
    """Run the complete analysis pipeline."""
    # Step 1: Define protein sets
    protein_pairs = get_protein_pairs()
    multi_domain, single_domain = get_multi_domain_proteins()
    all_proteins = list(set([p for pair in protein_pairs for p in pair] + multi_domain + single_domain))
    
    # Step 2: Generate embeddings for all proteins
    print("Generating embeddings for all proteins...")
    embedding_results = generate_embeddings_for_proteins(all_proteins)
    
    # Step 3: Analyze RMSD vs embedding distances for protein pairs
    print("Analyzing RMSD vs embedding distances...")
    rmsd_vs_emb = compare_rmsd_vs_embedding_distance(protein_pairs, embedding_results)
    rmsd_vs_emb.to_csv(os.path.join(RESULTS_DIR, "rmsd_vs_embedding.csv"), index=False)
    
    # Create scatter plots for RMSD vs embedding distances
    plt.figure(figsize=(10, 8))
    for emb_type in [col for col in rmsd_vs_emb.columns if 'embedding' in col]:
        plt.scatter(rmsd_vs_emb['rmsd'], rmsd_vs_emb[emb_type], label=emb_type)
        
        # Calculate correlation
        correlation = rmsd_vs_emb[['rmsd', emb_type]].corr().iloc[0, 1]
        print(f"Correlation between RMSD and {emb_type}: {correlation:.3f}")
    
    plt.xlabel('RMSD (Å)')
    plt.ylabel('Embedding Distance')
    plt.title('RMSD vs Embedding Distance')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "rmsd_vs_embedding.png"))
    
    # Step 4: Analyze mutation impact
    print("Analyzing mutation impact...")
    mutation_impact = analyze_mutation_impact(protein_pairs, embedding_results)
    mutation_impact.to_csv(os.path.join(RESULTS_DIR, "mutation_impact.csv"), index=False)
    
    # Step 5: Analyze domain separation
    print("Analyzing domain separation...")
    # Create simple domain annotations (single vs multi-domain)
    domain_annotations = {pdb_id: "multi-domain" for pdb_id in multi_domain}
    domain_annotations.update({pdb_id: "single-domain" for pdb_id in single_domain})
    
    tsne_results = visualize_domain_separation(embedding_results, domain_annotations)
    
    # Plot t-SNE results
    for emb_type, df in tsne_results.items():
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='tsne_1', y='tsne_2', hue='domain_type')
        plt.title(f't-SNE Visualization of {emb_type}')
        plt.savefig(os.path.join(RESULTS_DIR, f"tsne_{emb_type}.png"))
    
    # Step 6: Test embedding disentanglement
    print("Testing embedding disentanglement...")
    perturb_functions = get_perturbation_functions()
    disentanglement_results = test_embedding_disentanglement(embedding_results, perturb_functions)
    disentanglement_results.to_csv(os.path.join(RESULTS_DIR, "disentanglement.csv"), index=False)
    
    # Plot top dimensions affected by each perturbation
    for emb_type in disentanglement_results['embedding_type'].unique():
        plt.figure(figsize=(12, 10))
        df_subset = disentanglement_results[disentanglement_results['embedding_type'] == emb_type]
        sns.boxplot(data=df_subset, x='perturbation', y='difference')
        plt.title(f'Impact of Perturbations on {emb_type}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"disentanglement_{emb_type}.png"))
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(rmsd_vs_emb, mutation_impact, tsne_results, disentanglement_results)
    
    print("Analysis complete! Results saved to:", RESULTS_DIR)

def generate_summary_report(rmsd_data, mutation_data, tsne_data, disentanglement_data):
    """Generate a summary report of the analysis results."""
    with open(os.path.join(RESULTS_DIR, "summary_report.md"), "w") as f:
        f.write("# Protein Embedding Models Comparative Analysis\n\n")
        
        # RMSD vs Embedding Distance
        f.write("## 1. RMSD vs Embedding Distance Correlation\n\n")
        f.write("This analysis measures how well the distance in embedding space correlates with structural similarity.\n\n")
        
        for emb_type in disentanglement_data['embedding_type'].unique():
                f.write(f"### {emb_type}\n\n")
                
                # Calculate mean difference by perturbation type
                mean_diffs = disentanglement_data[disentanglement_data['embedding_type'] == emb_type].groupby('perturbation')['difference'].mean().sort_values(ascending=False)
                
                f.write("Average impact by perturbation type:\n\n")
                for pert, mean_diff in mean_diffs.items():
                    f.write(f"* **{pert}**: {mean_diff:.5f}\n")
                
                f.write("\nTop 5 most affected dimensions:\n\n")
                top_dims = disentanglement_data[disentanglement_data['embedding_type'] == emb_type].groupby('dimension')['difference'].mean().sort_values(ascending=False).head(5)
                for dim, mean_diff in top_dims.items():
                    f.write(f"* Dimension {dim}: {mean_diff:.5f}\n")
                
                f.write("\n")
        
        # Final summary
        f.write("## Conclusion\n\n")
        f.write("This report provides a comparative analysis of protein embedding models in terms of:\n")
        f.write("1. Correlation with structural similarity (RMSD)\n")
        f.write("2. Sensitivity to mutations\n")
        f.write("3. Ability to separate protein domains\n")
        f.write("4. Disentanglement of structural features in embedding dimensions\n\n")
        
        f.write("For detailed results, refer to the CSV files and visualizations in the results directory.\n")

# ========================== PART 7: FoldX Integration ==========================

def download_foldx():
    """
    Instructions to download FoldX (this doesn't actually download it as it requires license).
    """
    print("FoldX is a licensed software that needs to be downloaded from http://foldxsuite.crg.eu/")
    print("After downloading FoldX, place the executable in your project directory.")
    print("Make sure the foldx binary is executable (chmod +x foldx on Unix-like systems).")
    return os.path.exists("./foldx")

def calculate_residue_importance_with_foldx(pdb_id, chain='A'):
    """
    Calculate residue importance using FoldX position scan.
    This performs an alanine scan to identify important residues.
    
    Args:
        pdb_id: PDB ID of the protein
        chain: Chain ID to analyze
    
    Returns:
        DataFrame with residue importance scores
    """
    # Check if FoldX is available
    if not download_foldx():
        print("FoldX not found. Skipping residue importance calculation.")
        return None
    
    # Download PDB if not available
    pdb_file = download_pdb(pdb_id)
    if not pdb_file:
        return None
    
    # Create FoldX config file for position scan
    config_file = "position_scan.cfg"
    with open(config_file, "w") as f:
        f.write(f"command=PositionScan\n")
        f.write(f"pdb={pdb_id}.pdb\n")
        f.write(f"positions={chain}\n")
        f.write(f"output-file={pdb_id}_scan\n")
    
    # Run FoldX
    try:
        subprocess.run(["./foldx", "-f", config_file], check=True)
        
        # Parse results
        result_file = f"{pdb_id}_scan_PS.fxout"
        if os.path.exists(result_file):
            results = []
            with open(result_file, "r") as f:
                for line in f:
                    if not line.startswith("#") and line.strip():
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            results.append({
                                "residue": parts[0],
                                "position": parts[1],
                                "ddG": float(parts[2])
                            })
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            return df
        else:
            print(f"FoldX result file not found: {result_file}")
            return None
    except Exception as e:
        print(f"Error running FoldX: {e}")
        return None

# ========================== PART 8: Extended Analysis ==========================

def analyze_domain_preservation(multi_domain_proteins, embedding_results):
    """
    Analyze how well embeddings preserve domain information across proteins.
    
    Args:
        multi_domain_proteins: Dictionary mapping PDB IDs to domain boundary information
        embedding_results: Dictionary with embedding results
    
    Returns:
        DataFrame with domain similarity analysis
    """
    results = []
    
    # For each pair of multi-domain proteins
    pdb_ids = list(multi_domain_proteins.keys())
    
    for i, pdb_id1 in enumerate(pdb_ids):
        for j in range(i+1, len(pdb_ids)):
            pdb_id2 = pdb_ids[j]
            
            # Skip if embeddings are not available
            if pdb_id1 not in embedding_results or pdb_id2 not in embedding_results:
                continue
            
            # Get shared domains
            domains1 = multi_domain_proteins[pdb_id1]
            domains2 = multi_domain_proteins[pdb_id2]
            
            shared_domains = set(domains1.keys()).intersection(set(domains2.keys()))
            
            if not shared_domains:
                continue
            
            # For each embedding type
            for emb_type in [k for k in embedding_results[pdb_id1].keys() if 'embedding' in k]:
                if embedding_results[pdb_id1][emb_type] is None or embedding_results[pdb_id2][emb_type] is None:
                    continue
                
                # Calculate overall protein similarity
                overall_similarity = 1 - cosine(embedding_results[pdb_id1][emb_type], embedding_results[pdb_id2][emb_type])
                
                # Add to results
                for domain in shared_domains:
                    results.append({
                        'pdb_id1': pdb_id1,
                        'pdb_id2': pdb_id2,
                        'embedding_type': emb_type,
                        'shared_domain': domain,
                        'overall_similarity': overall_similarity
                    })
    
    return pd.DataFrame(results)

def analyze_per_residue_embeddings(pdb_ids, embedding_results):
    """
    Analyze per-residue embeddings for specific proteins.
    This requires models that provide per-residue embeddings.
    
    Args:
        pdb_ids: List of PDB IDs to analyze
        embedding_results: Dictionary with embedding results (should include per-residue embeddings)
    
    Returns:
        Dictionary with per-residue embedding analysis
    """
    # Load ESM2 model which provides per-residue embeddings
    esm2_model, esm2_converter = load_esm2_model()
    
    results = {}
    
    for pdb_id in pdb_ids:
        # Download and extract sequence
        pdb_file = download_pdb(pdb_id)
        if not pdb_file:
            continue
        
        sequence = extract_sequence_from_pdb(pdb_file)
        
        # Get per-residue embeddings
        if esm2_model is not None and esm2_converter is not None:
            data = [(pdb_id, sequence)]
            batch_labels, batch_strs, batch_tokens = esm2_converter(data)
            
            with torch.no_grad():
                results_esm = esm2_model(batch_tokens, repr_layers=[33], return_contacts=True)
            
            # Extract per-residue embeddings
            token_representations = results_esm["representations"][33]
            per_residue_embeddings = token_representations[0, 1:len(sequence)+1].numpy()
            
            # Extract predicted contacts
            contacts = results_esm["contacts"].numpy()
            
            # Calculate pairwise distances between residues in embedding space
            emb_distances = squareform(pdist(per_residue_embeddings, metric='cosine'))
            
            results[pdb_id] = {
                'sequence': sequence,
                'per_residue_embeddings': per_residue_embeddings,
                'embedding_distances': emb_distances,
                'predicted_contacts': contacts
            }
    
    return results

def compare_with_predicted_structure(sequence, embedding_results):
    """
    Compare embeddings with structures predicted by AlphaFold or other models.
    This is a placeholder - actual implementation would require running AlphaFold locally
    or using an API service like ColabFold.
    
    Args:
        sequence: Protein sequence
        embedding_results: Dictionary with embedding results
    
    Returns:
        Analysis of embedding vs predicted structure
    """
    print("Note: This function is a placeholder for comparing embeddings with predicted structures.")
    print("To implement this fully, you'd need to:")
    print("1. Run AlphaFold locally or use ColabFold API")
    print("2. Extract structural features from the predicted structure")
    print("3. Correlate these features with embedding dimensions")
    
    # Placeholder for AlphaFold prediction
    predicted_structure = None
    predicted_plddt = None
    
    # For demonstration, return a mock result
    return {
        'sequence': sequence,
        'structure_prediction_available': False,
        'embedding_structure_correlation': None
    }

# ========================== PART 9: Interactive Analysis ==========================

def interactive_analysis_menu():
    """
    Provide an interactive menu for analysis.
    This allows for selecting specific analyses to run.
    """
    print("\n==== Structural Biology - Protein Embedding Analysis ====")
    print("1. Download PDB structures and generate embeddings")
    print("2. Analyze RMSD vs embedding distance correlation")
    print("3. Analyze mutation impact on embeddings")
    print("4. Analyze domain separation in embedding space")
    print("5. Test embedding disentanglement")
    print("6. Run complete analysis pipeline")
    print("7. Exit")
    
    choice = input("\nEnter your choice (1-7): ")
    
    if choice == '1':
        protein_pairs = get_protein_pairs()
        multi_domain, single_domain = get_multi_domain_proteins()
        all_proteins = list(set([p for pair in protein_pairs for p in pair] + multi_domain + single_domain))
        
        print(f"Downloading {len(all_proteins)} PDB structures and generating embeddings...")
        embedding_results = generate_embeddings_for_proteins(all_proteins)
        
        # Save results
        with open(os.path.join(RESULTS_DIR, "embedding_results.pkl"), "wb") as f:
            import pickle
            pickle.dump(embedding_results, f)
        
        print("Embeddings generated and saved!")
        
    elif choice == '2':
        # Load embeddings if available
        try:
            import pickle
            with open(os.path.join(RESULTS_DIR, "embedding_results.pkl"), "rb") as f:
                embedding_results = pickle.load(f)
        except:
            print("Embedding results not found. Please run option 1 first.")
            return
        
        protein_pairs = get_protein_pairs()
        print("Analyzing RMSD vs embedding distances...")
        rmsd_vs_emb = compare_rmsd_vs_embedding_distance(protein_pairs, embedding_results)
        rmsd_vs_emb.to_csv(os.path.join(RESULTS_DIR, "rmsd_vs_embedding.csv"), index=False)
        
        print("Analysis complete! Results saved to:", os.path.join(RESULTS_DIR, "rmsd_vs_embedding.csv"))
        
    elif choice == '3':
        # Load embeddings if available
        try:
            import pickle
            with open(os.path.join(RESULTS_DIR, "embedding_results.pkl"), "rb") as f:
                embedding_results = pickle.load(f)
        except:
            print("Embedding results not found. Please run option 1 first.")
            return
        
        protein_pairs = get_protein_pairs()
        print("Analyzing mutation impact...")
        mutation_impact = analyze_mutation_impact(protein_pairs, embedding_results)
        mutation_impact.to_csv(os.path.join(RESULTS_DIR, "mutation_impact.csv"), index=False)
        
        print("Analysis complete! Results saved to:", os.path.join(RESULTS_DIR, "mutation_impact.csv"))
        
    elif choice == '4':
        # Load embeddings if available
        try:
            import pickle
            with open(os.path.join(RESULTS_DIR, "embedding_results.pkl"), "rb") as f:
                embedding_results = pickle.load(f)
        except:
            print("Embedding results not found. Please run option 1 first.")
            return
        
        multi_domain, single_domain = get_multi_domain_proteins()
        
        # Create simple domain annotations
        domain_annotations = {pdb_id: "multi-domain" for pdb_id in multi_domain}
        domain_annotations.update({pdb_id: "single-domain" for pdb_id in single_domain})
        
        print("Analyzing domain separation...")
        tsne_results = visualize_domain_separation(embedding_results, domain_annotations)
        
        # Save results
        for emb_type, df in tsne_results.items():
            df.to_csv(os.path.join(RESULTS_DIR, f"tsne_{emb_type}.csv"), index=False)
        
        print("Analysis complete! Results saved to RESULTS_DIR")
        
    elif choice == '5':
        # Load embeddings if available
        try:
            import pickle
            with open(os.path.join(RESULTS_DIR, "embedding_results.pkl"), "rb") as f:
                embedding_results = pickle.load(f)
        except:
            print("Embedding results not found. Please run option 1 first.")
            return
        
        print("Testing embedding disentanglement...")
        perturb_functions = get_perturbation_functions()
        disentanglement_results = test_embedding_disentanglement(embedding_results, perturb_functions)
        disentanglement_results.to_csv(os.path.join(RESULTS_DIR, "disentanglement.csv"), index=False)
        
        print("Analysis complete! Results saved to:", os.path.join(RESULTS_DIR, "disentanglement.csv"))
        
    elif choice == '6':
        print("Running complete analysis pipeline...")
        run_complete_analysis()
        
    elif choice == '7':
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Please enter a number between 1 and 7.")
    
    # Show menu again
    interactive_analysis_menu()

# ========================== PART 10: Extended Perturbation Functions ==========================

def introduce_point_mutation(sequence, position, new_aa='A'):
    """
    Introduce a point mutation at a specific position.
    
    Args:
        sequence: Protein sequence
        position: 0-based position to mutate
        new_aa: New amino acid to introduce
    
    Returns:
        Mutated sequence
    """
    if position < 0 or position >= len(sequence):
        return sequence
    
    return sequence[:position] + new_aa + sequence[position+1:]

def perturb_secondary_structure(sequence, ss_type='helix'):
    """
    Perturb residues likely to be in a specific secondary structure.
    
    Args:
        sequence: Protein sequence
        ss_type: Secondary structure type ('helix', 'sheet', 'loop')
    
    Returns:
        Perturbed sequence
    """
    # Amino acid propensities for different secondary structures
    helix_prone = "AEFIKLM"
    sheet_prone = "FILVWY"
    loop_prone = "DNPGST"
    
    if ss_type == 'helix':
        target_aas = helix_prone
        replacement = 'P'  # Proline disrupts helices
    elif ss_type == 'sheet':
        target_aas = sheet_prone
        replacement = 'G'  # Glycine is flexible and can disrupt sheets
    else:  # loop
        target_aas = loop_prone
        replacement = 'A'  # Alanine has low flexibility
    
    perturbed = ""
    for aa in sequence:
        if aa in target_aas:
            perturbed += replacement
        else:
            perturbed += aa
    
    return perturbed

def get_extended_perturbation_functions():
    """Get extended set of perturbation functions."""
    basic_functions = get_perturbation_functions()
    
    # Add more perturbation functions
    extended_functions = {
        "helix_disruptor": lambda seq: perturb_secondary_structure(seq, 'helix'),
        "sheet_disruptor": lambda seq: perturb_secondary_structure(seq, 'sheet'),
        "loop_disruptor": lambda seq: perturb_secondary_structure(seq, 'loop'),
        "mid_point_mutation": lambda seq: introduce_point_mutation(seq, len(seq) // 2)
    }
    
    # Combine dictionaries
    return {**basic_functions, **extended_functions}

# ========================== PART 11: Main Function ==========================

def main():
    """Main function to run the analysis."""
    print("Protein Embedding Analysis Framework")
    print("====================================")
    print("This program performs comparative analysis of protein embedding models.")
    print("It compares embedding distances with structural similarity (RMSD),")
    print("analyzes how mutations affect embeddings, and tests domain separation")
    print("and disentanglement properties of embedding models.")
    print("\nOptions:")
    print("1. Run interactive menu")
    print("2. Run complete analysis pipeline")
    
    choice = input("\nEnter your choice (1-2): ")
    
    if choice == '1':
        interactive_analysis_menu()
    elif choice == '2':
        run_complete_analysis()
    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    main()