 # simple code
import cv2
import imutils

cam = cv2.VideoCapture(0)
haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")



while True:
    img = cam.read()[1]
 
    img = imutils.resize(img,500)
    gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haar.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
       
    cv2.imshow("",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()


''' # tkinter gui
import cv2
import imutils
import tkinter as tk
from PIL import Image, ImageTk

# Initialize video capture and face classifier
cam = cv2.VideoCapture(0)
haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Create a main window for the GUI
window = tk.Tk()
window.title("Face Detection GUI")

# Create a canvas to display the video frames
canvas = tk.Canvas(window, width=640, height=480)
canvas.pack()

# Define a function to update the canvas with a new frame
def update_frame():
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Convert the frame to a PhotoImage for display
    img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.img = img  # Keep a reference to the image to prevent garbage collection

    window.after(10, update_frame)  # Call update_frame again after 10 milliseconds

# Start the video capture and continuously update the canvas
update_frame()
window.mainloop()

# Release resources
cam.release()
cv2.destroyAllWindows()
'''
