import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import face_recognition
import threading
from queue import Queue

app = Flask(__name__)

# CAMERA CONFIGURATION
# For IP Webcam/DroidCam: Use format "http://IP:PORT/video"
CAMERAS = {
    0: "Bijesh's PC Camera",
    "http://192.168.1.97:4747/video": "Phone Camera"  # Change IP to your phone's IP
}

# For DroidCam, you might need: "http://IP:4747/mjpegfeed?640x480"
# For IP Webcam: "http://IP:8080/video"

nimgs = 10
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create necessary directories
for directory in ['Attendance', 'static', 'static/faces', 'static/found']:
    os.makedirs(directory, exist_ok=True)

if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time,Camera')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    """Fast face detection using Haar Cascade"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )
        return face_points
    except:
        return []

def identify_face(facearray, knn, threshold=0.5):
    """Optimized face identification"""
    # Convert to RGB
    if len(facearray.shape) == 2:
        facearray = cv2.cvtColor(facearray, cv2.COLOR_GRAY2RGB)
    elif facearray.shape[2] == 4:
        facearray = cv2.cvtColor(facearray, cv2.COLOR_BGRA2RGB)
    else:
        facearray = cv2.cvtColor(facearray, cv2.COLOR_BGR2RGB)
    
    # Resize for faster processing
    small_face = cv2.resize(facearray, (0, 0), fx=0.5, fy=0.5)
    
    face_locations = face_recognition.face_locations(small_face, model='hog')  # Use HOG (faster)
    if not face_locations:
        return None
    
    face_encodings = face_recognition.face_encodings(small_face, known_face_locations=face_locations)
    if not face_encodings:
        return None
    
    distances, indices = knn.kneighbors([face_encodings[0]], n_neighbors=1)
    if distances[0][0] > threshold:
        return None
    
    return knn.predict([face_encodings[0]])[0]

def train_model():
    """Train KNN model with face encodings"""
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    
    print('Training model...')
    for user in userlist:
        user_path = f'static/faces/{user}'
        for imgname in os.listdir(user_path):
            img_path = f'{user_path}/{imgname}'
            img = face_recognition.load_image_file(img_path)
            
            # Resize for faster encoding
            small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            encodings = face_recognition.face_encodings(small_img, model='hog')
            
            if len(encodings) > 0:
                faces.append(encodings[0])
                labels.append(user)
    
    if len(faces) == 0:
        print("No faces found for training!")
        return False
    
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=min(5, len(faces)))
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')
    print(f'Model trained with {len(faces)} face samples!')
    return True

def extract_attendance():
    """Get today's attendance records"""
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    return df['Name'], df['Roll'], df['Time'], df.get('Camera', ['N/A']*len(df)), len(df)

def add_attendance(name, camera_name):
    """Add attendance record with camera info"""
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},{camera_name}')

def getallusers():
    """Get all registered users"""
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    
    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)
    
    return userlist, names, rolls, len(userlist)

def open_camera(camera_id):
    """Open camera with proper configuration"""
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return None
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce lag
    
    return cap

class CameraThread(threading.Thread):
    """Thread for handling individual camera"""
    def __init__(self, camera_id, camera_name, knn, mode='attendance', search_user=None):
        threading.Thread.__init__(self)
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.knn = knn
        self.mode = mode
        self.search_user = search_user
        self.stop_flag = False
        self.found = False
        
    def run(self):
        cap = open_camera(self.camera_id)
        if cap is None:
            print(f"Failed to open camera: {self.camera_name}")
            return
        
        frame_count = 0
        process_every_n_frames = 5  # Process every 5th frame
        
        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame from {self.camera_name}")
                break
            
            frame_count += 1
            
            # Process only every Nth frame
            if frame_count % process_every_n_frames == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
                faces = extract_faces(small_frame)
                
                for (x, y, w, h) in faces:
                    # Scale back to original
                    x, y, w, h = int(x*2.5), int(y*2.5), int(w*2.5), int(h*2.5)
                    face_img = frame[y:y+h, x:x+w]
                    
                    identified_person = identify_face(face_img, self.knn)
                    
                    if self.mode == 'attendance':
                        if identified_person:
                            add_attendance(identified_person, self.camera_name)
                            color = (50, 200, 50)
                            text = identified_person
                        else:
                            color = (0, 0, 255)
                            text = "Unknown"
                    
                    elif self.mode == 'search':
                        if identified_person == self.search_user:
                            self.found = True
                            color = (0, 255, 0)
                            text = f"FOUND: {identified_person}"
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_path = f'static/found/{self.search_user}_{timestamp}_{self.camera_name.replace(" ", "_")}.jpg'
                            cv2.imwrite(save_path, frame)
                            
                            with open(f'Attendance/SearchLog-{datetoday}.csv', 'a') as f:
                                f.write(f'\n{self.search_user},{datetime.now().strftime("%H:%M:%S")},Found,{self.camera_name}')
                            
                            self.stop_flag = True
                        else:
                            color = (100, 100, 100)
                            text = identified_person if identified_person else "Unknown"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show camera name
            cv2.putText(frame, self.camera_name, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(f'{self.mode.title()} - {self.camera_name}', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_flag = True
                break
        
        cap.release()
        cv2.destroyWindow(f'{self.mode.title()} - {self.camera_name}')

@app.route('/')
def home():
    names, rolls, times, cameras, l = extract_attendance()
    userlist, _, _, _ = getallusers()
    return render_template('home.html', names=names, rolls=rolls, times=times, 
                          cameras=cameras, l=l, totalreg=totalreg(), 
                          datetoday2=datetoday2, userlist=userlist)

@app.route('/start', methods=['GET'])
def start():
    model_path = 'static/face_recognition_model.pkl'
    if not os.path.exists(model_path):
        names, rolls, times, cameras, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times,
                              cameras=cameras, l=l, totalreg=totalreg(), 
                              datetoday2=datetoday2,
                              mess='No trained model found. Please add a new face first.')
    
    knn = joblib.load(model_path)
    
    # Start threads for each camera
    threads = []
    for cam_id, cam_name in CAMERAS.items():
        thread = CameraThread(cam_id, cam_name, knn, mode='attendance')
        thread.start()
        threads.append(thread)
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    names, rolls, times, cameras, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times,
                          cameras=cameras, l=l, totalreg=totalreg(), 
                          datetoday2=datetoday2)

@app.route('/add', methods=['GET', 'POST'])
def add():
    """Add new user - ONLY uses laptop camera (camera 0)"""
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    os.makedirs(userimagefolder, exist_ok=True)
    
    # ONLY use camera 0 (laptop camera) for registration
    cap = open_camera(0)
    if cap is None:
        names, rolls, times, cameras, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times,
                              cameras=cameras, l=l, totalreg=totalreg(), 
                              datetoday2=datetoday2,
                              mess='❌ Cannot access laptop camera for registration!')
    
    i, j = 0, 0
    
    while i < nimgs:
        ret, frame = cap.read()
        if not ret:
            break
            
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Images: {i}/{nimgs}', (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if j % 5 == 0:
                name = f'{newusername}_{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y+h, x:x+w])
                i += 1
            j += 1
        
        cv2.imshow('Adding New User - Laptop Camera', frame)
        if cv2.waitKey(1) == 27:  # ESC to cancel
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print('Training model...')
    train_model()
    
    names, rolls, times, cameras, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times,
                          cameras=cameras, l=l, totalreg=totalreg(), 
                          datetoday2=datetoday2,
                          mess=f'✅ User {newusername} registered successfully!')

@app.route('/search', methods=['POST'])
def search_user():
    searchuser = request.form['searchuser']
    
    model_path = 'static/face_recognition_model.pkl'
    if not os.path.exists(model_path):
        return render_template('home.html', mess="No trained model found.")
    
    knn = joblib.load(model_path)
    
    # Start threads for each camera
    threads = []
    for cam_id, cam_name in CAMERAS.items():
        thread = CameraThread(cam_id, cam_name, knn, mode='search', search_user=searchuser)
        thread.start()
        threads.append(thread)
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    found = any(thread.found for thread in threads)
    
    names, rolls, times, cameras, l = extract_attendance()
    userlist, _, _, _ = getallusers()
    
    return render_template('home.html', names=names, rolls=rolls, times=times,
                          cameras=cameras, l=l, totalreg=totalreg(), 
                          datetoday2=datetoday2, userlist=userlist,
                          mess=f"{'✅ User Found!' if found else '❌ User Not Found!'}")

@app.route('/test-cameras', methods=['GET'])
def test_cameras():
    """Test endpoint to check which cameras are working"""
    results = {}
    
    for cam_id, cam_name in CAMERAS.items():
        cap = open_camera(cam_id)
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                results[cam_name] = f"✅ Working (Resolution: {frame.shape[1]}x{frame.shape[0]})"
            else:
                results[cam_name] = "❌ Cannot read frames"
            cap.release()
        else:
            results[cam_name] = "❌ Cannot open camera"
    
    return results

if __name__ == '__main__':
    app.run(debug=True)