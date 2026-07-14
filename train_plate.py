from ultralytics import YOLO

def main():
    model = YOLO('C:\Users\KMAG\Documents\GitHub\Traffic-Violation-AI-Models\PlateDetection\PlateDetection.pt')

    results = model.train(
        task='detect',
        data='C:\Users\USER\Documents\PHRoads\PlateDetectionD\data.yaml',
        imgsz=640,
        single_cls=False,
        model='C:\Users\KMAG\Documents\GitHub\Traffic-Violation-AI-Models\PlateDetection\PlateDetection.pt',
        device=0, 
    )

if __name__ == '__main__':
    main()
