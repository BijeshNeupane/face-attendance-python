import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

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

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    distances, indices = model.kneighbors(facearray, n_neighbors=1)

    # Threshold: if the distance is too large, mark as "Unknown"
    if distances[0][0] > 3000:  # You can tune this threshold
        return ["Unknown"]
    return model.predict(facearray)



def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
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

    # If model is missing, show message
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template(
            'home.html',
            names=names,
            rolls=rolls,
            times=times,
            l=l,
            totalreg=totalreg(),
            datetoday2=datetoday2,
            mess='There is no trained model in the static folder. Please add a new face to continue.'
        )

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (86, 32, 251), -1)

            # Prepare face for prediction
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]

            if identified_person != "Unknown":
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (x, y - 15),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 15),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Place frame into background image
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, times, l = extract_attendance()
    return render_template(
        'home.html',
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        totalreg=totalreg(),
        datetoday2=datetoday2
    )




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
    
    cap = cv2.VideoCapture(0)
    found = False
    os.makedirs('static/found', exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]

            if identified_person == searchuser:
                found = True
                # Draw rectangle & name on frame before saving
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, identified_person, (x, y - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                # Save the annotated frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f'static/found/{searchuser}_{timestamp}.jpg'
                cv2.imwrite(save_path, frame)

                # Log to CSV
                with open(f'Attendance/SearchLog-{datetoday}.csv', 'a') as f:
                    f.write(f'\n{searchuser},{datetime.now().strftime("%H:%M:%S")},Found')

                break  # stop checking other faces

        cv2.imshow('Search User', frame)

        # Stop if found or user presses 'q'
        if found or cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, times, l = extract_attendance()
    return render_template(
        'home.html',
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        totalreg=totalreg(),
        datetoday2=datetoday2,
        userlist=userlist,
        mess=f"{'User Found and Saved!' if found else 'User Not Found!'}"
    )


if __name__ == '__main__':
    app.run(debug=True)
