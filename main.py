import cv2
import os
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from playsound import playsound

detector = FaceMeshDetector(maxFaces=1)
df = pd.read_csv("C:/Users/HP/Desktop/FACE_DATA/Flag.csv")
password = df.loc[0, "Password"]
face_detector = cv2.CascadeClassifier('../Resources/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_data_dir_list = os.listdir("C:/Users/HP/Desktop/FACE_DATA")
df["Tot_Faces"] = df["Tot_Faces"].replace(df.loc[0, "Tot_Faces"], len(face_data_dir_list) - 1)
face_Number = df.loc[0, "Tot_Faces"]
k = 0
m = 0


def face_distance():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img, )

        if faces:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]
            w, _ = detector.findDistance(pointLeft, pointRight)
            W = 6.3
            f = 750
            d = (W * f) / w
            cvzone.putTextRect(img, f'Distance: {int(d) / 100}m',
                               (face[10][0] - 100, face[10][1] - 50),
                               scale=1.5)
            if d < 65:
                cap.release()
                cv2.destroyAllWindows()
                return 1

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath)
        imageNp = np.array(pilImage)
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_detector.detectMultiScale(imageNp)
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            ids.append(Id)
    return faceSamples, ids


def create_and_check_dir(path, facial_id):
    direct = os.path.dirname(path)
    if not os.path.exists(direct):
        os.makedirs(direct)
    else:
        shutil.rmtree("C:/Users/HP/Desktop/FACE_DATA/" + str(facial_id))
        os.makedirs(direct)


while True:
    k = 0
    m=0
    flag = face_distance()
    playsound('C:/Users/HP/Desktop/stand_still.mp3')
    if flag == 1:
        cap = cv2.VideoCapture(0)
        i = face_Number
        face_list = os.listdir("C:/Users/HP/Desktop/FACE_DATA")
        for element in face_list:
            if len(face_list) == 1:
                break
            recognizer.read("C:/Users/HP/Desktop/TRAINERS/" + str(element) + "_trainer.yml")
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                ids, conf = recognizer.predict(roi_gray)
                if conf < 50:
                    print("Welcome!")
                    playsound('C:/Users/HP/Desktop/welcome.mp3')
                    k = 1
                    m=1
                    break

            cv2.imshow('frame', img)
            i -= 1
            if i == 0:
                break
            if m == 1:
                break
        if k == 0 and face_Number > 0:
            print("Intrusion Detected!!!")
            playsound('C:/Users/HP/Desktop/siren.mp3')
        cap.release()
        cv2.destroyAllWindows()

    print("***********************")
    print("1. NEW REGISTRATION \n2. DELETE FACE\n3. EXIT")
    print("***********************")
    choice = int(input("Enter your choice: "))

    if choice == 1:
        vid_cam = cv2.VideoCapture(0)
        pass_check = int(input("Enter password: "))
        face_id = input("Enter your ID: ")
        count = 0
        if pass_check == password:
            create_and_check_dir("C:/Users/HP/Desktop/FACE_DATA/" + str(face_id) + "/", face_id)
            os.chdir("C:/Users/HP/Desktop/FACE_DATA/" + str(face_id))
            while True:
                _, image_frame = vid_cam.read()
                gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1
                    cv2.imwrite("User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
                    cv2.imshow('frame', image_frame)

                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                elif count >= 50:
                    print("Successfully Captured")
                    break
            print("Face Registered")

            facials, Ids = getImagesAndLabels("C:/Users/HP/Desktop/FACE_DATA/" + str(face_id))
            s = recognizer.train(facials, np.array(Ids))
            recognizer.write("C:/Users/HP/Desktop/TRAINERS/" + str(face_id) + "_trainer.yml")
            print("Successfully trained")
            face_Number += 1
            df["Tot_Faces"] = df["Tot_Faces"].replace(df.loc[0, "Tot_Faces"], face_Number)
        else:
            print("Invalid Password")
        vid_cam.release()
        cv2.destroyAllWindows()

    elif choice == 2:
        pass_check = int(input("Enter password: "))
        face_id = input("Enter your ID: ")
        if pass_check == password:
            shutil.rmtree("C:/Users/HP/Desktop/FACE_DATA/" + str(face_id))
            os.remove("C:/Users/HP/Desktop/TRAINERS/" + str(face_id) + "_trainer.yml")
            face_Number -= 1
            df["Tot_Faces"] = df["Tot_Faces"].replace(df.loc[0, "Tot_Faces"], face_Number)
            print("Face Data deleted")

    elif choice == 3:
        break

    else:
        print("Invalid choice!")
