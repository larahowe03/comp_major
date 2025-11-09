import yaml
from pathlib import Path
from tqdm import tqdm
import shutil

def remap_kaggle_for_contour_model(kaggle_yaml, output_dir):
    # Kaggle ID to Contour Model ID
    class_mapping = {
        0: 0,         # bishop (generic) to bishop
        1: 0,         # black-bishop to bishop
        2: 2,         # black-king to king
        3: 3,         # black-knight to knight
        4: 4,         # black-pawn to pawn
        5: 5,         # black-queen to queen
        6: 1,         # black-rook to castle
        7: 0,         # white-bishop to bishop
        8: 2,         # white-king to king
        9: 3,         # white-knight to knight
        10: 4,        # white-pawn to pawn
        11: 5,        # white-queen to queen
        12: 1,        # white-rook to castle
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
    
    contour_names = [
        'bishop',  
        'castle',    
        'king',      
        'knight',    
        'pawn',      
        'queen'      
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
        'files_processed': 0
    }
    
    for split in splits:
        if split not in config:
            continue
        
        # Get directories
        img_dir = kaggle_dir / 'images' / split
        label_dir = kaggle_dir / 'labels' / split
                
        # Create output directories
        output_img_dir = output_path / 'images' / split
        output_label_dir = output_path / 'labels' / split
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
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
                    stats['total_annotations'] += 1
                    
                    old_class_id = int(parts[0])
                    
                    # Check if class should be remapped
                    if old_class_id in class_mapping:
                        new_class_id = class_mapping[old_class_id]
                        
                        # Remap
                        parts[0] = str(new_class_id)
                        remapped_lines.append(' '.join(parts) + '\n')
                        stats['remapped_annotations'] += 1
            
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
        'names': contour_names
    }
    
    if 'test' in config:
        new_config['test'] = 'images/test'
    
    output_yaml = output_path / 'data.yaml'
    with open(output_yaml, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    
if __name__ == "__main__":
    # Configuration
    kaggle_yaml = "datasets/kaggle_dataset_warped_remapped_contours/data.yaml"
    output_dir = "datasets/kaggle_dataset_warped_remapped_contours_"
    
    remap_kaggle_for_contour_model(kaggle_yaml, output_dir)