import cv2
import numpy as np
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS


class ObjectDetection:

    def __init__(self):
        self.model = "MobileNetSSD_deploy.caffemodel"
        self.prototxt = "MobileNetSSD_deploy.prototxt.txt"
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

    def define_classes(self):
        """Define classes and associates colors randomly"""

        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def run_detection(self):

        print("Start object detection ...")

        self.define_classes()
        vs = WebcamVideoStream(0).start()
        fps = FPS().start()

        while True:

            frame = vs.read()
            (h, w) = frame.shape[:-1]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         scalefactor=0.007843, size=(300, 300), mean=127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            # Loop over detected objects
            for i in np.arange(0, detections.shape[2]):

                # Get proba for each object
                probability = detections[0, 0, i, 2]

                # Set up threshold for filtering detection
                if probability > 0.5:
                    # Get prediction index
                    index = int(detections[0, 0, i, 1])

                    # Get bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    label = "{}: {:.2f}%".format(self.classes[index], probability * 100)

                    y = startY - 10 if startY - 10 > 10 else startY + 10

                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  self.colors[index], 2)
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[index], 2)

            cv2.imshow('frame', frame)
            fps.update()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                vs.stop()
                fps.stop()
                break

        print("Fps: {:.2f}".format(fps.fps()))
        fps.update()


if __name__ == '__main__' :
    detector = ObjectDetection()
    detector.run_detection()
