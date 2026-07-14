from ultralytics import YOLO

def main():
    model = YOLO('C:\Users\KMAG\Documents\GitHub\Traffic-Violation-AI-Models\ViolationDetection\ViolationDetection.pt')

    results = model.train(
        task='detect',
        data='C:\Users\USER\Documents\PHRoads\ViolationDetection\data.yaml',
        imgsz=768,
        single_cls=False,
        model='C:\Users\KMAG\Documents\GitHub\Traffic-Violation-AI-Models\ViolationDetection\ViolationDetection.pt',
        device=0, 
    )

if __name__ == '__main__':
    main()
