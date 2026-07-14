from ultralytics import YOLO

def main():
    model = YOLO('C:\Users\KMAG\Documents\GitHub\Traffic-Violation-AI-Models\TricycleBodyNumberDetection\BodyNumber.pt')

    results = model.train(
        task='detect',
        data='C:\Users\USER\Documents\PHRoads\TricycleBodyNumberDetectionB\data.yaml',
        imgsz=640,
        single_cls=False,
        model='C:\Users\KMAG\Documents\GitHub\Traffic-Violation-AI-Models\TricycleBodyNumberDetection\BodyNumber.pt',
        device=0, 
    )

if __name__ == '__main__':
    main()
