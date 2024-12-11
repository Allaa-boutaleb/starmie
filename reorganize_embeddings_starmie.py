import os
import shutil
import logging
import pickle
import argparse
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def verify_file_integrity(src_path, dst_path):
    """Verify files have same size and can be loaded"""
    if os.path.getsize(src_path) != os.path.getsize(dst_path):
        return False
    try:
        with open(src_path, 'rb') as f:
            pickle.load(f)
        with open(dst_path, 'rb') as f:
            pickle.load(f)
        return True
    except:
        return False

def reorganize_embeddings(benchmark):
    """Reorganize embeddings for a given benchmark"""
    logging.info(f"Processing benchmark: {benchmark}")
    
    # Setup paths
    vectors_dir = Path("vectors")
    benchmark_dir = vectors_dir / benchmark
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    # First copy the benchmark file
    benchmark_file = None
    for ext in ['.pkl', '.pickle']:
        src_path = Path(f"data/{benchmark}/{benchmark}UnionBenchmark{ext}")
        if src_path.exists():
            benchmark_file = src_path
            break
    
    if benchmark_file:
        dst_path = benchmark_dir / f"benchmark.pkl"
        if not dst_path.exists():
            logging.info(f"Copying benchmark file: {benchmark_file} to {dst_path}")
            shutil.copy2(benchmark_file, dst_path)
            
            if verify_file_integrity(benchmark_file, dst_path):
                logging.info(f"Successfully verified benchmark file: {dst_path}")
            else:
                logging.error(f"Benchmark file verification failed: {dst_path}")
                os.remove(dst_path)
    else:
        logging.warning(f"No benchmark file found for {benchmark}")
    
    # First handle query embeddings (only need original)
    src_dir = Path(f"data/{benchmark}/vectors/query")
    if src_dir.exists():
        pkl_files = list(src_dir.glob("*.pkl"))
        if pkl_files:
            src_path = pkl_files[0]
            dst_path = benchmark_dir / "starmie_query_embeddings.pkl"
            
            if not dst_path.exists():
                logging.info(f"Copying {src_path} to {dst_path}")
                shutil.copy2(src_path, dst_path)
                
                if verify_file_integrity(src_path, dst_path):
                    logging.info(f"Successfully verified: {dst_path}")
                else:
                    logging.error(f"File verification failed: {dst_path}")
                    os.remove(dst_path)
        else:
            logging.warning(f"No query embeddings found in: {src_dir}")
    
    # Then handle datalake embeddings for each variant
    variants = ['original', 'p-col']
    for variant in variants:
        src_base = f"data/{benchmark if variant == 'original' else benchmark + '-' + variant}/vectors"
        src_dir = Path(src_base) / 'datalake'
        
        if not src_dir.exists():
            logging.warning(f"Source directory not found: {src_dir}")
            continue
        
        pkl_files = list(src_dir.glob("*.pkl"))
        if not pkl_files:
            logging.warning(f"No .pkl files found in: {src_dir}")
            continue
            
        src_path = pkl_files[0]
        dst_name = f"starmie_{variant}_datalake_embeddings.pkl"
        dst_path = benchmark_dir / dst_name
        
        if dst_path.exists():
            logging.info(f"File already exists, skipping: {dst_path}")
            continue
            
        logging.info(f"Copying {src_path} to {dst_path}")
        shutil.copy2(src_path, dst_path)
        
        if verify_file_integrity(src_path, dst_path):
            logging.info(f"Successfully verified: {dst_path}")
        else:
            logging.error(f"File verification failed: {dst_path}")
            os.remove(dst_path)

def main():
    parser = argparse.ArgumentParser(description='Reorganize embeddings into a centralized structure')
    parser.add_argument('benchmark', type=str, help='Benchmark name (e.g., santos, tus)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    
    args = parser.parse_args()
    setup_logging()
    reorganize_embeddings(args.benchmark)

if __name__ == '__main__':
    main()