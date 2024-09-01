import tkinter as tk
from tkinter import messagebox
from tkinter import Message, Text
import cv2, os
import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()
print("LBPHFaceRecognizer is available")
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Create a Tkinter window
window = tk.Tk()
window.title("PresencePro")
window.geometry('2688x1536')

# Create a custom style
style = ttk.Style()
style.configure("Curvy.TLabel", borderwidth=2, relief="groove")

# Adding Background Image
bg_image = Image.open("Girl.png")
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(window, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

# Adding Images
def load_image(path, width, height):
    img = Image.open(path)
    img = img.resize((width, height), Image.LANCZOS)
    return ImageTk.PhotoImage(img)

# Load and place images
img_computer = load_image("Designer.png", 100, 100)
img_label = tk.Label(window, image=img_computer, bg="black")
img_label.place(x=1300, y=20)

img_computer2 = load_image("Student.png", 100, 100)
img_label2 = tk.Label(window, image=img_computer2, bg="#36454F",relief="sunken",borderwidth=10,)
img_label2.place(x=50, y=25)

# Create the label with the custom style
message = tk.Label(
    window,
    text="PresencePro: Automated Attendance System",
    bg="#36454F",
    fg="#ffd700",
    width=50,
    height=2,
    font=("Comic Sans MS", 30, "bold"),
    relief="groove",
    borderwidth=10,
)

message.place(x=200, y=20)


lbl = tk.Label(
    window,
    text="Enter ID   :",
    width=20,
    height=1,
    fg="#ffd700",
    bg="#36454F",
    font=("Comic Sans MS", 15, " bold "),
    relief="groove",
    borderwidth=10,
)
lbl.place(x=400, y=200)

txt = tk.Entry(
    window, width=20, bg="#EEE8AA", fg="blue", font=("Comic Sans MS", 15, " bold "),relief="solid",
    borderwidth=5,
)
txt.place(x=700, y=205)

lbl2 = tk.Label(
    window,
    text="Enter Name :",
    width=20,
    fg="#ffd700",
    bg="#36454F",
    height=1,
    font=("Comic Sans MS", 15, " bold "),relief="groove",
    borderwidth=10,
)
lbl2.place(x=400, y=260)

txt2 = tk.Entry(
    window, width=20, bg="#EEE8AA", fg="blue", font=("Comic Sans MS", 15, " bold "),relief="solid",
    borderwidth=5,
)
txt2.place(x=700, y=265)

lbl5 = tk.Label(
    window,
    text="Enter Mail  :",
    width=20,
    fg="#ffd700",
    bg="#36454F",
    height=1,
    font=("Comic Sans MS", 15, " bold "),relief="groove",
    borderwidth=10,
)
lbl5.place(x=400, y=320)

txt3 = tk.Entry(
    window, width=20, bg="#EEE8AA", fg="blue", font=("Comic Sans MS", 15, " bold "),relief="solid",
    borderwidth=5,
)
txt3.place(x=700, y=325)

lbl3 = tk.Label(
    window,
    text="Notification:",
    width=20,
    fg="black",
    bg="#ffd700",
    height=2,
    font=("Comic Sans MS", 15, " bold "),relief="groove",
    borderwidth=10,
)
lbl3.place(x=400, y=400)

message = tk.Label(
    window,
    text="",
    bg="#36454F",
    fg="#ffd700",
    width=30,
    height=2,
    activebackground="black",
    font=("Comic Sans MS", 15, " bold "),relief="groove",
    borderwidth=10,
)
message.place(x=700, y=400)

lbl3 = tk.Label(
    window,
    text="Attendance:",
    width=20,
    fg="#ffd700",
    bg="#36454F",
    height=2,
    font=("Comic Sans MS", 15, " bold "),relief="groove",
    borderwidth=10,
)
lbl3.place(x=400, y=650)

message2 = tk.Label(
    window,
    text="",
    fg="#ffd700",
    bg="#36454F",
    activeforeground="green",
    width=30,
    font=("Comic Sans MS", 15, " bold "),relief="groove",
    borderwidth=10,
)
message2.place(height=120, x=700, y=650)


def clear():
    txt.delete(0, "end")
    res = ""
    message.configure(text=res)


def clear2():
    txt2.delete(0, "end")
    res = ""
    message.configure(text=res)


def clear3():
    txt3.delete(0, "end")
    res = ""
    message.configure(text=res)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata

        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def TakeImages():
    Id = (txt.get())
    name = (txt2.get())
    mail = (txt3.get())
    if ((name.isalpha()) or (" " in name)):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                cv2.imwrite(
                    "TrainingImage/" + name + "." + Id + "." + str(sampleNum) + ".jpg",
                    gray[y : y + h, x : x + w],
                )
                cv2.imshow("frame", img)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id + " Name : " + name
        row = [Id, name, mail]
        with open("StudentDetails/StudentDetails.csv", "a+") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        message.configure(text=res)
    else:
        if is_number(Id):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if name.isalpha():
            res = "Enter Numeric Id"
            message.configure(text=res)

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    res = "Image Trained"
    message.configure(text=res)


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert("L")
        imageNp = np.array(pilImage, "uint8")
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails/StudentDetails.csv")
    print(df.columns)
    print(df)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ["Id", "Name", "Date", "Time"]
    attendance = pd.DataFrame(columns=col_names)
    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])        #conf: confidence                           
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id=' '                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    Hour, Minute, Second = timeStamp.split(":")
    fileName = (
        "Attendance/Attendance_"
        + date
        + "_"
        + Hour
        + "-"
        + Minute
        + "-"
        + Second
        + ".csv"
    )
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    res = attendance
    message2.configure(text=res)


clearButton = tk.Button(
    window,
    text="Clear",
    command=clear,
    fg="#ffd700",
    bg="#36454F",
    width=20,
    height=1,
    activebackground="#ffd700",
    font=("Comic Sans MS", 10, " bold "),relief="raised",
    borderwidth=5,
)
clearButton.place(x=980, y=205)
clearButton2 = tk.Button(
    window,
    text="Clear",
    command=clear2,
    fg="#ffd700",
    bg="#36454F",
    width=20,
    height=1,
    activebackground="#ffd700",
    font=("Comic Sans MS", 10, " bold "),relief="raised",
    borderwidth=5,
)
clearButton2.place(x=980, y=265)
clearButton3 = tk.Button(
    window,
    text="Clear",
    command=clear3,
    fg="#ffd700",
    bg="#36454F",
    width=20,
    height=1,
    activebackground="#ffd700",
    font=("Comic Sans MS", 10, " bold "),relief="raised",
    borderwidth=5,
)
clearButton3.place(x=980, y=325)

takeImg = tk.Button(
    window,
    text="Scan Facial Features",
    command=TakeImages,
    fg="#000080",
    bg="#ffd700",
    width=20,
    height=3,
    activebackground="#ffd700",
    font=("Comic Sans MS", 15, " bold "),relief="groove",
    borderwidth=10,
)
takeImg.place(x=200, y=500)
trainImg = tk.Button(
    window,
    text="Train Images",
    command=TrainImages,
    fg="#000080",
    bg="#ffd700",
    width=20,
    height=3,
    activebackground="#ffd700",
    font=("Comic Sans MS", 15, " bold "),relief="groove",
    borderwidth=10,
)
trainImg.place(x=500, y=500)
trackImg = tk.Button(
    window,
    text="Take Attendance",
    command=TrackImages,
    fg="#000080",
    bg="#ffd700",
    width=20,
    height=3,
    activebackground="#ffd700",
    font=("Comic Sans MS", 15, " bold "),relief="groove",
    borderwidth=10,
)
trackImg.place(x=800, y=500)
quitWindow = tk.Button(
    window,
    text="Quit",
    command=window.destroy,
    fg="#000080",
    bg="#ffd700",
    width=20,
    height=3,
    activebackground="#ffd700",
    font=("Comic Sans MS", 15, " bold "),relief="groove",
    borderwidth=10,
)
quitWindow.place(x=1100, y=500)

window.mainloop()