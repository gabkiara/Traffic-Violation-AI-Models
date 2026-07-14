import json
import os
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

def reengineer_model(checkpoint_path, output_script="train_plate.py"):
    print(f"Loading weights from: {checkpoint_path}...\n")
    model = YOLO(checkpoint_path)
    
    # 1. Extract the training arguments
    train_args = {}
    if hasattr(model, 'ckpt') and isinstance(model.ckpt, dict) and 'args' in model.ckpt:
        train_args = model.ckpt['args']
    elif hasattr(model, 'overrides') and model.overrides:
        train_args = model.overrides
    else:
        print("❌ Error: Could not find training metadata in this .pt file.")
        return

    if isinstance(train_args, str):
        try:
            train_args = json.loads(train_args)
        except Exception:
            pass

    # 2. Load official YOLOv8 defaults to fill in the blanks
    defaults = vars(get_cfg())

    # 3. Augmentation keys we want to inspect
    aug_keys = [
        'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 
        'shear', 'perspective', 'flipud', 'fliplr', 'bgr', 'mosaic', 
        'mixup', 'copy_paste', 'auto_augment', 'erasing', 'crop_fraction'
    ]

    print("=========================================")
    print("DETECTED AUGMENTATION PARAMETERS")
    print("=========================================")
    for k in aug_keys:
        # Check if it was explicitly overwritten, otherwise grab the default value used
        if k in train_args:
            print(f"✨ {k}: {train_args[k]} (Explicitly set)")
        elif k in defaults:
            print(f"📦 {k}: {defaults[k]} (YOLOv8 Default)")
    print("=========================================\n")

    # 4. Clean up system-specific arguments for the file generation
    ignore_keys = {
        'save_dir', 'project', 'name', 'exist_ok', 'pretrained', 
        'verbose', 'device', 'rect', 'resume', 'nosave', 'noval', 
        'plots', 'conf', 'iou', 'max_det', 'half', 'dnn', 'source'
    }
    
    cleaned_args = {k: v for k, v in train_args.items() if k not in ignore_keys}
    model_task = cleaned_args.pop('task', 'detect')
    
    base_model = train_args.get('model', 'yolov8n.pt')
    if base_model.endswith('.pt') and not os.path.exists(base_model):
        base_model = os.path.basename(base_model)

    # 5. Format the arguments into a Python dictionary string
    dict_lines = []
    for key, value in cleaned_args.items():
        if isinstance(value, str):
            dict_lines.append(f"        {key}='{value}',")
        else:
            dict_lines.append(f"        {key}={value},")
            
    args_string = "\n".join(dict_lines)

    # 6. Generate the train.py script content
    script_content = f"""from ultralytics import YOLO

def main():
    model = YOLO('{base_model}')

    results = model.train(
        task='{model_task}',
{args_string}
        device=0, 
    )

if __name__ == '__main__':
    main()
"""

    with open(output_script, "w") as f:
        f.write(script_content)
        
    print(f"✅ Success! Reengineered training file saved as: {output_script}")

if __name__ == "__main__":
    # Using the raw string format to prevent Windows path errors
    reengineer_model(r"C:\Users\KMAG\Documents\GitHub\Traffic-Violation-AI-Models\PlateDetection\PlateDetection.pt")