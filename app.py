import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, session, send_file
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import face_recognition
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# multiprocessing imports
import multiprocessing
from multiprocessing import Process, Queue, Manager

app = Flask(__name__)
app.secret_key = 'replace_this_with_a_strong_secret'  # <-- change this

# ------------------------- CONFIG & GLOBALS -------------------------
CAMERAS = {
    0: "Bijesh's PC Camera",
    "http://192.168.1.97:4747/video": "Phone Camera"  # Change IP to your phone's IP
}

nimgs = 10
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for directory in ['Attendance', 'static', 'static/faces', 'static/found']:
    os.makedirs(directory, exist_ok=True)

model_path = 'static/face_recognition_model.pkl'
USER_DB = 'users.db'
USER_BASEPATH = 'static/faces'

# ------------------------- SQLITE USER DB -------------------------
def init_db():
    con = sqlite3.connect(USER_DB)
    cur = con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT,
                    fullname TEXT
                )''')
    con.commit()
    con.close()

init_db()

# ------------------------- AUTH HELPERS -------------------------
def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapped

def get_user_folder(username=None):
    if username is None:
        username = session.get('user')
    folder = os.path.join('Attendance', username)
    os.makedirs(folder, exist_ok=True)
    return folder

def user_today_csv(username=None):
    folder = get_user_folder(username)
    path = os.path.join(folder, f'Attendance-{datetoday}.csv')
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('Name,Roll,Time,Camera\n')
    return path

def user_searchlog_path(username=None):
    folder = get_user_folder(username)
    path = os.path.join(folder, f'SearchLog-{datetoday}.csv')
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('Name,Time,Status,Camera\n')
    return path

# ------------------------- ORIGINAL HELPERS (kept & adapted) -------------------------
def totalreg():
    try:
        return len(os.listdir(USER_BASEPATH))
    except FileNotFoundError:
        return 0

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )
        return face_points
    except Exception:
        return []

def identify_face(facearray, knn, threshold=0.5):
    try:
        if len(facearray.shape) == 2:
            facearray = cv2.cvtColor(facearray, cv2.COLOR_GRAY2RGB)
        elif facearray.shape[2] == 4:
            facearray = cv2.cvtColor(facearray, cv2.COLOR_BGRA2RGB)
        else:
            facearray = cv2.cvtColor(facearray, cv2.COLOR_BGR2RGB)

        small_face = cv2.resize(facearray, (0, 0), fx=0.5, fy=0.5)
        face_locations = face_recognition.face_locations(small_face, model='hog')
        if not face_locations:
            return None

        face_encodings = face_recognition.face_encodings(small_face, known_face_locations=face_locations)
        if not face_encodings:
            return None

        distances, indices = knn.kneighbors([face_encodings[0]], n_neighbors=1)
        if distances[0][0] > threshold:
            return None

        return knn.predict([face_encodings[0]])[0]
    except Exception:
        return None

def train_model():
    faces = []
    labels = []
    userlist = os.listdir(USER_BASEPATH) if os.path.exists(USER_BASEPATH) else []

    print('Training model...')
    for user in userlist:
        user_path = f'{USER_BASEPATH}/{user}'
        if not os.path.isdir(user_path):
            continue
        for imgname in os.listdir(user_path):
            img_path = f'{user_path}/{imgname}'
            try:
                img = face_recognition.load_image_file(img_path)
            except Exception:
                continue

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
    joblib.dump(knn, model_path)
    print(f'Model trained with {len(faces)} face samples!')
    return True

def extract_attendance(username=None):
    csv_path = user_today_csv(username)
    df = pd.read_csv(csv_path)
    return df['Name'], df['Roll'], df['Time'], df.get('Camera', ['N/A'] * len(df)), len(df)

def add_attendance(name, camera_name, username=None):
    if username is None:
        username = session.get('user')
    username_from_label = name.split('_')[0] if '_' in name else name
    userid = name.split('_')[1] if '_' in name else name
    current_time = datetime.now().strftime("%H:%M:%S")

    csv_path = user_today_csv(username)
    df = pd.read_csv(csv_path)
    try:
        if int(userid) not in list(df['Roll'].astype(int)):
            with open(csv_path, 'a') as f:
                f.write(f'\n{username_from_label},{userid},{current_time},{camera_name}')
    except Exception:
        if userid not in list(df['Roll'].astype(str)):
            with open(csv_path, 'a') as f:
                f.write(f'\n{username_from_label},{userid},{current_time},{camera_name}')

def getallusers_original():
    userlist = os.listdir(USER_BASEPATH) if os.path.exists(USER_BASEPATH) else []
    names = []
    rolls = []
    for i in userlist:
        if '_' in i:
            name, roll = i.split('_', 1)
        else:
            name, roll = i, ''
        names.append(name)
        rolls.append(roll)
    return userlist, names, rolls, len(userlist)

def open_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap

# --------------------- Multiprocessing-based camera + recognizer ---------------------

def camera_process(camera_id, camera_name, req_queue, res_queue, stop_flag, process_every_n_frames=5):
    """
    Runs in separate process.
    - captures frames,
    - sends encoded JPEG frames to req_queue when it's time to process,
    - receives detection results from res_queue and overlays them on display_frame,
    - shows window and allows 'q' to quit (sets stop_flag).
    """
    cap = open_camera(camera_id)
    if cap is None:
        print(f"[camera_process] Cannot open camera {camera_name}")
        return

    frame_count = 0
    consecutive_failures = 0
    max_failures = 30

    try:
        while not stop_flag['stop']:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"[{camera_name}] Too many read failures, exiting.")
                    break
                continue
            consecutive_failures = 0
            frame_count += 1
            display_frame = frame.copy()

            # send for recognition every Nth frame
            if frame_count % process_every_n_frames == 0:
                # encode as JPEG to reduce IPC size
                success, encoded = cv2.imencode('.jpg', frame)
                if success:
                    try:
                        req_queue.put_bytes(encoded.tobytes())  # using raw bytes method if available
                    except Exception:
                        # fallback: put bytes into queue normally
                        req_queue.put(encoded.tobytes())

            # non-blocking read of results for this camera
            try:
                while True:
                    result = res_queue.get_nowait()
                    # result is list of detections: [(x, y, w, h, label), ...] in original frame coordinates
                    for det in result:
                        x, y, w, h, label = det
                        color = (50, 200, 50) if label and not label == "Unknown" else (0, 0, 255)
                        text = label if label else "Unknown"
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(display_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception:
                # no result available
                pass

            # show camera name
            cv2.putText(display_frame, camera_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(f'Camera - {camera_name}', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag['stop'] = True
                break
    except Exception as e:
        print(f"[camera_process] Error ({camera_name}): {e}")
    finally:
        cap.release()
        cv2.destroyWindow(f'Camera - {camera_name}')
        print(f"[camera_process] {camera_name} closed")

def recognizer_process(all_req_queues, all_res_queues, shared_stop, username_for_attendance, mode_shared):
    """
    Runs in single process. Loads KNN model and performs face detection/recognition.
    all_req_queues: dict camera_name -> Queue
    all_res_queues: dict camera_name -> Queue
    shared_stop: Manager().dict() with {'stop': False}
    username_for_attendance: the username to log attendance/search results for
    mode_shared: Manager().dict with {'mode': 'attendance' or 'search', 'search_user': None}
    """
    # Load model once
    if not os.path.exists(model_path):
        print("[recognizer] No model found; exiting recognizer.")
        return

    knn = joblib.load(model_path)
    print("[recognizer] Model loaded in recognizer process.")

    while not shared_stop['stop']:
        # iterate over request queues
        for cam_key, q in list(all_req_queues.items()):
            try:
                frame_bytes = q.get_nowait()
            except Exception:
                continue
            try:
                # decode bytes -> numpy image
                nparr = np.frombuffer(frame_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                # do face detection with Haar first (faster)
                faces = extract_faces(img)
                detections = []
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        # crop face and run identification on the face patch
                        face_img = img[y:y+h, x:x+w]
                        label = identify_face(face_img, knn)
                        detections.append((x, y, w, h, label))

                        # handle logging for attendance/search
                        if mode_shared['mode'] == 'attendance':
                            if label:
                                add_attendance(label, cam_key, username=username_for_attendance)
                        elif mode_shared['mode'] == 'search':
                            search_target = mode_shared.get('search_user')
                            if label == search_target:
                                search_log = user_searchlog_path(username_for_attendance)

                                # ensure file exists and has header if empty
                                if not os.path.exists(search_log):
                                    with open(search_log, 'w') as f:
                                        f.write("Name,Time,Status,Camera\n")

                                # always append a new search entry even if user already exists
                                with open(search_log, 'a') as f:
                                    f.write(f"{search_target},{datetime.now().strftime('%H:%M:%S')},Found,{cam_key}\n")

                                print(f"[recognizer] ✅ {search_target} found in {cam_key}")
                                shared_stop['stop'] = True
                                break


                # return detections to camera-specific response queue
                res_q = all_res_queues.get(cam_key)
                if res_q:
                    res_q.put(detections)
            except Exception as e:
                print(f"[recognizer] error processing frame from {cam_key}: {e}")

    print("[recognizer] stopping recognizer process")

# ------------------------- FLASK ROUTES (original + auth) -------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    mess = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        con = sqlite3.connect(USER_DB)
        cur = con.cursor()
        cur.execute("SELECT password FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        con.close()

        if row and check_password_hash(row[0], password):
            session['user'] = username
            return redirect(url_for('home'))
        else:
            mess = 'Invalid username or password.'

    return render_template('login.html', mess=mess)

@app.route('/register', methods=['GET', 'POST'])
def register():
    mess = ''
    if request.method == 'POST':
        username = request.form['username']
        password_raw = request.form['password']
        fullname = request.form.get('fullname', '')

        hashed = generate_password_hash(password_raw)
        try:
            con = sqlite3.connect(USER_DB)
            cur = con.cursor()
            cur.execute("INSERT INTO users (username, password, fullname) VALUES (?, ?, ?)",
                        (username, hashed, fullname))
            con.commit()
            con.close()
            os.makedirs(os.path.join('Attendance', username), exist_ok=True)
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            mess = 'Username already exists.'

    return render_template('register.html', mess=mess)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    user = session['user']
    try:
        names, rolls, times, cameras, l = extract_attendance(user)
    except Exception:
        csv = user_today_csv(user)
        df = pd.read_csv(csv)
        names, rolls, times = df['Name'], df['Roll'], df['Time']
        cameras, l = df.get('Camera', ['N/A'] * len(df)), len(df)

    userlist, _, _, _ = getallusers_original()
    return render_template('home.html', names=names, rolls=rolls, times=times,
                           cameras=cameras, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, userlist=userlist, username=user)

@app.route('/start', methods=['GET'])
@login_required
def start():
    # spawn one recognizer + camera processes in attendance mode
    if not os.path.exists(model_path):
        names, rolls, times, cameras, l = extract_attendance(session['user'])
        return render_template('home.html', names=names, rolls=rolls, times=times,
                              cameras=cameras, l=l, totalreg=totalreg(),
                              datetoday2=datetoday2,
                              mess='No trained model found. Please add a new face first.')

    manager = Manager()
    shared_stop = manager.dict()
    shared_stop['stop'] = False
    mode_shared = manager.dict()
    mode_shared['mode'] = 'attendance'
    mode_shared['search_user'] = None

    # prepare queues: one req & res queue per camera (keyed by cam_name)
    all_req_queues = {}
    all_res_queues = {}
    cam_processes = []

    for cam_id, cam_name in CAMERAS.items():
        req_q = Queue(maxsize=5)
        res_q = Queue(maxsize=5)
        all_req_queues[cam_name] = req_q
        all_res_queues[cam_name] = res_q

        p = Process(target=camera_process, args=(cam_id, cam_name, req_q, res_q, shared_stop))
        p.start()
        cam_processes.append(p)

    recognizer = Process(target=recognizer_process, args=(all_req_queues, all_res_queues, shared_stop, session['user'], mode_shared))
    recognizer.start()

    # wait for all camera processes to finish (they set shared_stop on 'q' or recognizer finds target)
    try:
        for p in cam_processes:
            p.join()
    except KeyboardInterrupt:
        shared_stop['stop'] = True

    # terminate recognizer
    shared_stop['stop'] = True
    recognizer.join(timeout=2)
    try:
        recognizer.terminate()
    except Exception:
        pass

    names, rolls, times, cameras, l = extract_attendance(session['user'])
    return render_template('home.html', names=names, rolls=rolls, times=times,
                          cameras=cameras, l=l, totalreg=totalreg(),
                          datetoday2=datetoday2)

@app.route('/add', methods=['GET', 'POST'])
@login_required
def add():
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        userimagefolder = f'static/faces/{newusername}_{newuserid}'
        os.makedirs(userimagefolder, exist_ok=True)

        cap = open_camera(0)
        if cap is None:
            names, rolls, times, cameras, l = extract_attendance(session['user'])
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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Images: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if j % 5 == 0:
                    name = f'{newusername}_{i}.jpg'
                    cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y + h, x:x + w])
                    i += 1
                j += 1

            cv2.imshow('Adding New User - Laptop Camera', frame)
            if cv2.waitKey(1) == 27:  # ESC to cancel
                break

        cap.release()
        cv2.destroyAllWindows()

        print('Training model...')
        train_model()

        names, rolls, times, cameras, l = extract_attendance(session['user'])
        return render_template('home.html', names=names, rolls=rolls, times=times,
                              cameras=cameras, l=l, totalreg=totalreg(),
                              datetoday2=datetoday2,
                              mess=f'✅ User {newusername} registered successfully!')

    return redirect(url_for('home'))

@app.route('/search', methods=['POST'])
@login_required
def search_user():
    searchuser = request.form['searchuser']

    if not os.path.exists(model_path):
        return render_template('home.html', mess="No trained model found.")

    manager = Manager()
    shared_stop = manager.dict()
    shared_stop['stop'] = False
    mode_shared = manager.dict()
    mode_shared['mode'] = 'search'
    mode_shared['search_user'] = searchuser

    all_req_queues = {}
    all_res_queues = {}
    cam_processes = []

    for cam_id, cam_name in CAMERAS.items():
        req_q = Queue(maxsize=5)
        res_q = Queue(maxsize=5)
        all_req_queues[cam_name] = req_q
        all_res_queues[cam_name] = res_q

        p = Process(target=camera_process, args=(cam_id, cam_name, req_q, res_q, shared_stop))
        p.start()
        cam_processes.append(p)

    recognizer = Process(target=recognizer_process, args=(all_req_queues, all_res_queues, shared_stop, session['user'], mode_shared))
    recognizer.start()

    import time
    start_time = time.time()
    timeout = 300  # 5 minutes

    while not shared_stop['stop'] and time.time() - start_time < timeout:
        alive = any(p.is_alive() for p in cam_processes)
        if not alive:
            break
        time.sleep(0.1)

    shared_stop['stop'] = True
    time.sleep(0.5)

    for p in cam_processes:
        try:
            p.terminate()
        except Exception:
            pass

    try:
        recognizer.terminate()
    except Exception:
        pass

    cv2.destroyAllWindows()

    found = False
    # check search log for any 'Found' entries for today
    log_path = user_searchlog_path(session['user'])
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            content = f.read()
            if 'Found' in content:
                found = True

    names, rolls, times, cameras, l = extract_attendance(session['user'])
    userlist, _, _, _ = getallusers_original()

    return render_template('home.html', names=names, rolls=rolls, times=times,
                           cameras=cameras, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, userlist=userlist,
                           mess=f"{'✅ User Found!' if found else '❌ User Not Found!'}")

@app.route('/test-cameras', methods=['GET'])
@login_required
def test_cameras():
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

@app.route('/download/attendance')
@login_required
def download_attendance():
    username = session['user']
    file_path = user_today_csv(username)
    return send_file(file_path, as_attachment=True)

@app.route('/download/searchlog')
@login_required
def download_searchlog():
    username = session['user']
    file_path = user_searchlog_path(username)
    return send_file(file_path, as_attachment=True)

# ------------------------- MAIN -------------------------
if __name__ == '__main__':
    # necessary on Windows to avoid fork issues
    try:
        multiprocessing.set_start_method('spawn')
    except Exception:
        pass

    os.makedirs('Attendance', exist_ok=True)
    os.makedirs(USER_BASEPATH, exist_ok=True)
    train_model()
    app.run(debug=True)
