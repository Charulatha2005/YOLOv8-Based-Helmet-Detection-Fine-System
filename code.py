import cv2
from ultralytics import YOLO

# models
person_model = YOLO("yolov8n.pt")
helmet_model = YOLO(r"Code\models\helmet delection\helmet.pt")

cap = cv2.VideoCapture(r"Code\models\helmet delection\ttt.mp4")

PERSON = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = person_model(frame)
    person_id = 1

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls == PERSON:
                px1,py1,px2,py2 = map(int,box.xyxy[0])
                head_y2 = int(py1 + (py2 - py1) * 0.4)
                head_crop = frame[py1:head_y2, px1:px2]
                helmet_found = False
                nohelmet_found = False
                helmet_results = helmet_model(head_crop)

                for hr in helmet_results:
                    for hbox in hr.boxes:
                        hcls = int(hbox.cls[0])

                        if hcls == 0:
                            helmet_found = True
                        else:
                            nohelmet_found = True

                if nohelmet_found:
                    color = (0,0,255)
                    label = f"Person {person_id} NO HELMET - FINE Rs500"

                elif helmet_found:
                    color = (0,255,0)
                    label = f"Person {person_id} Helmet OK"

                else:
                    color = (0,165,255)
                    label = f"Person {person_id}  No Helmet fine INR 500"

                cv2.rectangle(frame,(px1,py1),(px2,py2),color,2)

                cv2.putText(frame,
                            label,
                            (px1,py1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2)

                cv2.rectangle(frame,(px1,py1),(px2,head_y2),(255,255,0),2)

                person_id += 1

    cv2.imshow("Helmet Fine System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
