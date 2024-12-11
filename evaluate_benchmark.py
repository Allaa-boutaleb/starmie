import os
import json
import pickle
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine, euclidean
from checkPrecisionRecall import calcMetrics, loadDictionaryFromPickleFile
from naive_search import NaiveSearcher
from tqdm import tqdm
import pandas as pd

def setup_directories(benchmark):
    """Create output directory structure for the benchmark"""
    base_dir = Path(f"output/{benchmark}")
    variants = ['original','p-col']
    
    for variant in variants:
        (base_dir / variant).mkdir(parents=True, exist_ok=True)
    
    return base_dir

def load_embeddings(benchmark, variant):
    """Load embeddings for a specific benchmark variant"""
    base_path = Path(f"vectors/{benchmark}")
    
    try:
        query_path = base_path / "starmie_query_embeddings.pkl"
        datalake_path = base_path / f"starmie_{variant}_datalake_embeddings.pkl"
        
        queries = loadDictionaryFromPickleFile(query_path)
        datalake = loadDictionaryFromPickleFile(datalake_path)
        
        return queries, datalake
    except FileNotFoundError as e:
        print(f"Warning: Could not load embeddings for {benchmark}/{variant}: {e}")
        return None, None

def load_table_structure(table_path):
    """Load CSV table to get column names and order"""
    try:
        df = pd.read_csv(table_path)
        return list(df.columns)
    except:
        return None

def calculate_detailed_similarity_metrics(original_embeddings, variant_embeddings, data_dir, variant):
    """Calculate column-level similarity metrics between original and variant embeddings"""
    detailed_metrics = {"tables": []}
    
    # Convert lists of tuples to dictionaries for easier lookup
    orig_dict = {entry[0]: entry[1] for entry in original_embeddings}
    var_dict = {entry[0]: entry[1] for entry in variant_embeddings}
    
    # Extract benchmark name from data_dir
    benchmark = str(data_dir).split('/')[1]
    
    for table_id in orig_dict:
        if table_id not in var_dict:
            continue
            
        # Construct paths for original and variant tables
        orig_path = Path("data") / benchmark / "datalake" / table_id
        var_path = Path("data") / f"{benchmark}-{variant}" / "datalake" / table_id
        
        orig_columns = load_table_structure(orig_path)
        var_columns = load_table_structure(var_path)
        
        if not orig_columns or not var_columns:
            continue
            
        # Get column embeddings
        orig_embeddings_table = orig_dict[table_id]
        var_embeddings_table = var_dict[table_id]
        
        if len(orig_embeddings_table) != len(var_embeddings_table):
            continue
            
        # Calculate column-wise similarities
        column_similarities = []
        euclidean_distances = []
        cosine_similarities = []
        
        # Create mappings for both original and variant
        orig_col_map = {col_name: (idx, emb) for idx, (col_name, emb) in enumerate(zip(orig_columns, orig_embeddings_table))}
        var_col_map = {col_name: (idx, emb) for idx, (col_name, emb) in enumerate(zip(var_columns, var_embeddings_table))}
        
        # Compare columns by name
        for col_name in orig_col_map:
            if col_name in var_col_map:
                orig_idx, orig_emb = orig_col_map[col_name]
                var_idx, var_emb = var_col_map[col_name]
                
                curr_euclidean = euclidean(orig_emb, var_emb)
                curr_cosine = 1 - cosine(orig_emb, var_emb)
                
                column_similarities.append({
                    "column_name": col_name,
                    "original_position": orig_idx,
                    "permuted_position": var_idx,
                    "euclidean_distance": float(curr_euclidean),
                    "cosine_similarity": float(curr_cosine)
                })
                
                euclidean_distances.append(curr_euclidean)
                cosine_similarities.append(curr_cosine)
        
        table_metrics = {
            "table_name": table_id,
            "num_columns": len(orig_columns),
            "column_similarities": column_similarities,
            "aggregate_metrics": {
                "mean_euclidean": float(np.mean(euclidean_distances)),
                "mean_cosine": float(np.mean(cosine_similarities))
            }
        }
        
        detailed_metrics["tables"].append(table_metrics)
    
    return detailed_metrics

def evaluate_benchmark(benchmark_name, distances_only=False):
    """Main evaluation function for a benchmark"""
    # Parameters from run_tus_all.py and test_naive_search.py
    params = {
        'santos': {
            'max_k': 10, 
            'k_range': 1, 
            'sample_size': None, 
            'threshold': 0.1,
            'scale': 1.0,
            'encoder': 'cl',
            'matching': 'exact',
            'table_order': 'column'
        },
        'pylon': {
            'max_k': 10, 
            'k_range': 1, 
            'sample_size': None, 
            'threshold': 0.1,
            'scale': 1.0,
            'encoder': 'cl',
            'matching': 'exact',
            'table_order': 'column'
        },
        'tus': {
            'max_k': 60, 
            'k_range': 10, 
            'sample_size': 150, 
            'threshold': 0.1,
            'scale': 1.0,
            'encoder': 'cl',
            'matching': 'exact',
            'table_order': 'column'
        },
        'tusLarge': {
            'max_k': 60, 
            'k_range': 10, 
            'sample_size': 100, 
            'threshold': 0.1,
            'scale': 1.0,
            'encoder': 'cl',
            'matching': 'exact',
            'table_order': 'column'
        }
    }[benchmark_name]
    
    output_dir = setup_directories(benchmark_name)
    base_path = Path(f"vectors/{benchmark_name}")
    data_path = Path(f"data/{benchmark_name}")
    gt_path = base_path / "benchmark.pkl"
    
    if distances_only:
        variants = ['original','p-col']
        for variant in variants:
            try:
                original_path = base_path / "starmie_original_datalake_embeddings.pkl"
                variant_path = base_path / f"starmie_{variant}_datalake_embeddings.pkl"
                
                original_datalake = loadDictionaryFromPickleFile(original_path)
                variant_datalake = loadDictionaryFromPickleFile(variant_path)
                
                detailed_metrics = calculate_detailed_similarity_metrics(
                    original_datalake, 
                    variant_datalake,
                    data_path,
                    variant
                )
                
                # Save detailed metrics
                with open(output_dir / variant / 'raw_distances.json', 'w') as f:
                    json.dump(detailed_metrics, f, indent=2)
                    
                print(f"Updated detailed metrics for {variant}")
                
            except Exception as e:
                print(f"Error processing {variant}: {e}")
        return

    # variants = ['original', 'p-col']
    variants = ['original','p-col']

    results = {}
    
    for variant in variants:
        datalake_path = base_path / f"starmie_{variant}_datalake_embeddings.pkl"
        query_path = base_path / "starmie_query_embeddings.pkl"
        
        if not datalake_path.exists() or not query_path.exists():
            print(f"Skipping {variant}: embeddings not found")
            continue
        
        # Do the search first
        searcher = NaiveSearcher(str(datalake_path), scale=params['scale'])
        returnedResults = {}
        unionability_scores = {}  # New temporary dictionary to store scores
        
        # Load queries and sample if needed
        queries = loadDictionaryFromPickleFile(query_path)
        queries.sort(key=lambda x: x[0])
        
        # Sample queries for tus and tusLarge
        if params['sample_size'] is not None:
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(queries), size=params['sample_size'], replace=False)
            queries = [queries[i] for i in indices]
        
        print(f"\nProcessing {variant} variant:")
        for query in tqdm(queries, desc="Processing queries", unit="query"):
            # Get both scores and table names from topk
            search_results = searcher.topk(
                enc=params['encoder'],
                query=query,
                K=params['max_k'], 
                threshold=params['threshold']
            )
            
            # Store table names for traditional metrics
            returnedResults[query[0]] = [r[1] for r in search_results]
            
            # Store the unionability scores separately
            unionability_scores[query[0]] = [float(r[0]) for r in search_results]
        
        # Calculate traditional metrics
        metrics = calcMetrics(
            max_k=params['max_k'],
            k_range=params['k_range'],
            resultFile=returnedResults,
            gtPath=gt_path,
            record=False,
            verbose=False
        )
        
        # Add unionability scores to the per_query_metrics
        for query_id in metrics['per_query_metrics']:
            if query_id in unionability_scores:
                metrics['per_query_metrics'][query_id]['unionability_scores'] = \
                    unionability_scores[query_id]
        
        # Save detailed metrics
        with open(output_dir / variant / 'detailed_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        results[variant] = metrics['system_metrics']

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate benchmark and calculate similarity metrics")
    parser.add_argument("benchmark", 
                       choices=['santos', 'tus', 'tusLarge', 'pylon'],
                       help="Benchmark to evaluate")
    parser.add_argument("--distances_only", 
                       action="store_true",
                       help="Only recalculate detailed distance metrics without redoing evaluation")
    parser.add_argument("--data_dir",
                       type=str,
                       default=None,
                       help="Optional: Override default data directory path")
    
    args = parser.parse_args()
    
    # Update data path if provided
    if args.data_dir:
        data_path = Path(args.data_dir)
    
    evaluate_benchmark(args.benchmark, args.distances_only)