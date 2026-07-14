from ultralytics import YOLO

def main():
    model = YOLO('C:\Users\KMAG\Documents\GitHub\Traffic-Violation-AI-Models\ColorDetection\ColorDetection.pt')

    results = model.train(
        task='detect',
        data='data.yaml',
        imgsz=512,
        single_cls=False,
        model='C:\Users\KMAG\Documents\GitHub\Traffic-Violation-AI-Models\ColorDetection\ColorDetection.pt',
        device=0, 
    )

if __name__ == '__main__':
    main()
