import cv2
import random

#thres = 0.45 # Threshold to detect object

classNames = []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(480,480)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):

    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                if (draw):
                    detecbox.append(box)
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                    cv2.putText(img,str(round(confidence*100,2))+"%",(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
           
            else:
                # detecbox.append(0, "test")
                cv2.rectangle(img,box,color=(0,0,255),thickness=2)
                cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                cv2.putText(img,str(round(confidence*100,2))+"%",(box[0]+200,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)

    return img, detecbox

def draw_output(img):
    cv2.putText(img,"Suhu : "+str(suhu)+" C",(460,30), 
            cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,0),1)
    cv2.putText(img,"Tekanan : "+str(press)+" Psi",(460,50),
            cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,0),1)
    cv2.line(img, (320,0), (320, 480), (255,0,0), 1)
    cv2.line(img, (0,240), (640, 240), (255,0,0), 1)

    return img

def movement(box):
    x, y = box[0][0], box[0][1]
    # x_start, y_start, x_end, y_end = box[0][0], box[0][1], box[0][2], box[0][3]
    if x < 213:
        cv2.putText(img,"Movement: Kiri",(40,40),
        cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
    elif x > 427:
        cv2.putText(img,"Movement: Kanan",(40,40),
        cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
    elif x > 213 and x < 427:
        cv2.putText(img,"Movement: Maju",(40,40),
        cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)

    # width = x_end-x_start
    # height = y_end-y_start
    # return width, height
  
if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)

    record = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (640, 480))

    while True:
        suhu = random.randint(21, 24)
        press = random.randint(120, 140)
        success, img = cap.read()
        detecbox = []
        
        result, box = getObjects(img,0.45,0.2, objects=['bottle'])
        draw_output(img)
        if len(box) != 0:
            movement(box)
            # print(box)
        record.write(img)
        cv2.imshow("Output",img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    record.release()
    cv2.destroyAllWindows()