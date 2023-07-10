import cv2
import numpy as np
import time #for calculate frame


vedio_path = '#./video.mp4' #route of video
min_confidence = 0.5

def detectAndDisplay(frame):
    start_time = time.time()
    img = cv2.resize(frame, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape
    #cv2.imshow("Original Image", img)

    #window size setting
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    #Predicting class of detected object 
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #Put the desired class id / starting 0 
            if class_id == 0 and confidence > min_confidence:
                #boxing detected object
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
               
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}".format(classes[class_ids[i]], confidences[i]*100)
            print(i, label)
            color = colors[i] #Set for box line color /If you want to use one color, use (255,255,255) (B,G,R)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    cv2.imshow("YOLO test", img)
    
#yolo format and call class id
model_file = './yolov5.weights' #CHANGE IT TO SUIT YOUR DEVELOPMENT ENVIRONMENT
config_file = './yolov5.cfg' #CHANGE IT TO SUIT YOUR DEVELOPMENT ENVIRONMENT
net = cv2.dnn.readNet(model_file, config_file)

#USE 'GPU'
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#Class(file of name) open / CHANGE IT TO SUIT YOUR DEVELOPMENT ENVIRONMENT
classes = []
with open("./label", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Run Video
cap = cv2.VideoCapture(0) #IF YOU DON'T USE WEBCAM, CHANGE 0 INTO 'video path'
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    #When you input 'q', it means quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()