import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import face_recognition

cameraName = "Bijesh's PC"  # Change to your camera name

app = Flask(__name__)

nimgs = 10

imgBackground=cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray, knn, threshold=0.5):
    # Ensure RGB
    if len(facearray.shape) == 2 or facearray.shape[2] == 1:
        facearray = cv2.cvtColor(facearray, cv2.COLOR_GRAY2RGB)
    else:
        facearray = cv2.cvtColor(facearray, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(facearray)
    if not face_locations:
        return None

    face_encodings = face_recognition.face_encodings(facearray, known_face_locations=face_locations)
    if not face_encodings:
        return None

    distances, indices = knn.kneighbors([face_encodings[0]], n_neighbors=1)
    if distances[0][0] > threshold:
        return None  # Unknown person
    predicted_name = knn.predict([face_encodings[0]])[0]
    return predicted_name


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img_path = f'static/faces/{user}/{imgname}'
            img = face_recognition.load_image_file(img_path)
            
            encodings = face_recognition.face_encodings(img)
            if len(encodings) > 0:  # Only if a face is found
                faces.append(encodings[0])  # 128D vector
                labels.append(user)

    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    userlist, _, _, _ = getallusers()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, userlist=userlist)

@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    # Check model
    model_path = 'static/face_recognition_model.pkl'
    if not os.path.exists(model_path):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2,
                               mess='No trained model found. Please add a new face first.')

    knn = joblib.load(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_template('home.html', mess="⚠ Could not access webcam!")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:  # skip every other frame
            continue

        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        faces = extract_faces(small_frame)

        for (x, y, w, h) in faces:
            # scale back to original
            x, y, w, h = x*2, y*2, w*2, h*2
            face_img = frame[y:y+h, x:x+w]
            identified_person = identify_face(face_img, knn)

            if identified_person:
                add_attendance(identified_person)
                color = (50, 50, 255)
                text = identified_person
            else:
                color = (0, 0, 255)
                text = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)




@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/search', methods=['POST'])
def search_user():
    searchuser = request.form['searchuser']
    userlist, _, _, _ = getallusers()

    model_path = 'static/face_recognition_model.pkl'
    if not os.path.exists(model_path):
        return render_template('home.html', mess="No trained model found.")

    knn = joblib.load(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_template('home.html', mess="⚠ Could not access webcam!")

    found = False
    os.makedirs('static/found', exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        faces = extract_faces(small_frame)

        for (x, y, w, h) in faces:
            x, y, w, h = x*2, y*2, w*2, h*2
            face_img = frame[y:y+h, x:x+w]
            identified_person = identify_face(face_img, knn)

            if identified_person == searchuser:
                found = True
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
                cv2.putText(frame, identified_person, (x, y-10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), 2)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f'static/found/{searchuser}_{timestamp}.jpg'
                cv2.imwrite(save_path, frame)

                with open(f'Attendance/SearchLog-{datetoday}.csv', 'a') as f:
                    f.write(f'\n{searchuser},{datetime.now().strftime("%H:%M:%S")},Found,{cameraName}')
                break

        cv2.imshow('Search User', frame)
        if found or cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2,
                           userlist=userlist,
                           mess=f"{'User Found and Saved!' if found else 'User Not Found!'}")


if __name__ == '__main__':
    app.run(debug=True)
