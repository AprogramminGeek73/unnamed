import face_recognition
import  os
import cv2
import numpy as np

path = 'Images/'
images = []
className = []
p_name = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])#removing extensions 

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistKnown = findEncodings(images)
# print(len(encodelistKnown))


cap = cv2.VideoCapture(0)

for p in range(6):
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        if len(facesCurFrame) == 1:
            encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

            for encodeFace, Faceloc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodelistKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodelistKnown, encodeFace)
                print(faceDis)
                print(matches)
                
                matchIndex = np.argmin(faceDis)
                print(matchIndex)
                if matches[matchIndex]:
                    name = className[matchIndex].upper()
                    print(name)
                    p_name.append(name[0].lower())
        else:
            continue
        break

d_b = p_name.count('d')

if d_b < 4:
    print("absent")
else:
    print("present")

print(p_name)

    # cv2.imshow("Current Frame", imgS)
    





