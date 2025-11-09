import yaml
from pathlib import Path
from tqdm import tqdm
import shutil

def remap_kaggle_to_model_classes(kaggle_yaml, output_dir, backup=True):    
    # Kaggle ID to Your Model ID
    class_mapping = {
        0: None,      # bishop (generic) - SKIP (ambiguous which color)
        1: 0,         # black-bishop to black_bishop
        2: 2,         # black-king to black_king
        3: 3,         # black-knight to black_knight
        4: 4,         # black-pawn to black_pawn
        5: 5,         # black-queen to black_queen
        6: 1,         # black-rook to black_castle
        7: 6,         # white-bishop to white_bishop
        8: 8,         # white-king to white_king
        9: 9,         # white-knight to white_knight
        10: 10,       # white-pawn to white_pawn
        11: 11,       # white-queen to white_queen
        12: 7,        # white-rook to white_castle
    }
        
    kaggle_names = {
        0: 'bishop (generic)',
        1: 'black-bishop',
        2: 'black-king',
        3: 'black-knight',
        4: 'black-pawn',
        5: 'black-queen',
        6: 'black-rook',
        7: 'white-bishop',
        8: 'white-king',
        9: 'white-knight',
        10: 'white-pawn',
        11: 'white-queen',
        12: 'white-rook'
    }
    
    model_names = [
        'black_bishop', 
        'black_castle',    
        'black_king',      
        'black_knight',    
        'black_pawn',      
        'black_queen',     
        'white_bishop',    
        'white_castle',    
        'white_king',      
        'white_knight',    
        'white_pawn',      
        'white_queen'      
    ]
        
    # Load Kaggle config
    with open(kaggle_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    kaggle_dir = Path(kaggle_yaml).parent
    output_path = Path(output_dir)
    
    # Process each split
    splits = ['train', 'val', 'test']
    stats = {
        'total_annotations': 0,
        'remapped_annotations': 0,
        'skipped_annotations': 0,
        'files_processed': 0
    }
    
    for split in splits:
        if split not in config:
            continue
        
        # Get directories
        img_dir = kaggle_dir / config[split]
        label_dir = Path(str(img_dir).replace('images', 'labels'))
        
        if not img_dir.exists() or not label_dir.exists():
            continue
        
        # Create output directories
        output_img_dir = output_path / 'images' / split
        output_label_dir = output_path / 'labels' / split
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image and label files
        image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.bmp'))
                
        for img_path in tqdm(image_files, desc=f"Remapping {split}"):
            # Copy image
            output_img_path = output_img_dir / img_path.name
            shutil.copy2(img_path, output_img_path)
            
            # Process label
            label_path = label_dir / (img_path.stem + '.txt')
            
            if not label_path.exists():
                continue
            
            # Read and remap annotations
            remapped_lines = []
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        stats['total_annotations'] += 1
                        
                        old_class_id = int(parts[0])
                        
                        # Check if class should be remapped
                        if old_class_id in class_mapping:
                            new_class_id = class_mapping[old_class_id]
                            
                            if new_class_id is None:
                                # Skip this annotation
                                stats['skipped_annotations'] += 1
                                continue
                            
                            # Remap
                            parts[0] = str(new_class_id)
                            remapped_lines.append(' '.join(parts) + '\n')
                            stats['remapped_annotations'] += 1
                        else:
                            pass
            
            # Write remapped labels
            output_label_path = output_label_dir / (img_path.stem + '.txt')
            with open(output_label_path, 'w') as f:
                f.writelines(remapped_lines)
            
            stats['files_processed'] += 1
    
    # Create new data.yaml
    new_config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': model_names
    }
    
    if 'test' in config:
        new_config['test'] = 'images/test'
    
    output_yaml = output_path / 'data.yaml'
    with open(output_yaml, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)

if __name__ == "__main__":
    kaggle_yaml = "datasets/kaggle_dataset_warped/data.yaml"
    output_dir = "datasets/kaggle_dataset_warped_remapped"
    
    response = input("\nContinue? (yes/no): ").strip().lower()
    