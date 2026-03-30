
"""
Clean up a HuggingFace repository by deleting all non-inference files and squashing history.
Usage:
    python huggingface/clean_hf_repo.py --repo username/phobert-cccd-ner --token YOUR_HF_TOKEN
"""

import argparse
import sys
from huggingface_hub import HfApi

def clean_repo(repo_id: str, token: str):
    api = HfApi(token=token)
    
    print(f"==> Cleaning repository: {repo_id}")
    
    # 1. List all files to identify junk
    try:
        files = api.list_repo_files(repo_id, repo_type="model")
    except Exception as e:
        print(f"[ERROR] Could not access repo: {e}")
        sys.exit(1)
        
    junk_patterns = [
        "checkpoint-", 
        "optimizer.pt", 
        "scheduler.pt", 
        "trainer_state.json", 
        "training_args.bin",
        "runs/",
        "logs/"
    ]
    
    to_delete = []
    for f in files:
        for pattern in junk_patterns:
            if pattern in f:
                to_delete.append(f)
                break
                
    if not to_delete:
        print("[INFO] No junk files found in the repository.")
    else:
        print(f"[INFO] Found {len(to_delete)} files/folders to delete.")
        for f in to_delete:
            try:
                api.delete_file(path_in_repo=f, repo_id=repo_id, token=token)
                print(f"  Deleted: {f}")
            except Exception as e:
                print(f"  Failed to delete {f}: {e}")

    # 2. Squash history to reclaim space
    print("[INFO] Squashing repository history to reclaim space...")
    try:
        api.super_squash_history(
            repo_id=repo_id,
            repo_type="model",
            commit_message="chore: clean up and squash history",
            token=token
        )
        print("[SUCCESS] History squashed. Repository should be much smaller now.")
    except Exception as e:
        print(f"[WARNING] Could not squash history: {e}")

    print(f"\n[DONE] Check your repo at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean HF Repo")
    parser.add_argument("--repo", required=True, help="HF repo ID")
    parser.add_argument("--token", required=True, help="HF API token")
    args = parser.parse_args()
    clean_repo(args.repo, args.token)
