from ultralytics import YOLO

def main():
    model = YOLO('C:\Users\KMAG\Documents\GitHub\Traffic-Violation-AI-Models\VehicleDetection\VehicleDetection.pt')

    results = model.train(
        task='detect',
        data='data.yaml',
        imgsz=832,
        single_cls=False,
        model='C:\Users\KMAG\Documents\GitHub\Traffic-Violation-AI-Models\VehicleDetection\VehicleDetection.pt',
        device=0, 
    )

if __name__ == '__main__':
    main()
