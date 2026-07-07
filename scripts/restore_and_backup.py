import os
import shutil

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BACKUP_DIR = os.path.join(DATA_DIR, "backup")

# Original line limits (including headers)
ORIGINAL_LIMITS = {
    "users.csv": 10001,
    "orders.csv": 20001,
    "order_items.csv": 43526,
    "events.csv": 80001,
    "reviews.csv": 15002,
    "products.csv": 2001
}

def main():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f"📁 Created backup directory at {BACKUP_DIR}")

    for filename, limit in ORIGINAL_LIMITS.items():
        filepath = os.path.join(DATA_DIR, filename)
        backup_path = os.path.join(BACKUP_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️ File {filename} not found, skipping.")
            continue
            
        print(f"📦 Extracting original {limit} lines from {filename} to backup...")
        
        # Read the exact original lines
        original_lines = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                original_lines.append(line)
        
        # Write to backup
        with open(backup_path, 'w', encoding='utf-8', newline='') as f:
            f.writelines(original_lines)
            
        # Overwrite the active file in data/ to restore it
        shutil.copyfile(backup_path, filepath)
        print(f"✅ Restored and backed up {filename} successfully.")

    print("\n🎉 Restore and backup completed! All files in data/ are now reset to original Kaggle data.")

if __name__ == "__main__":
    main()
