import cv2
import os
import json
from flask import Flask, request, render_template, redirect, url_for, session, send_file
from datetime import date, datetime
import numpy as np
import face_recognition
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# multiprocessing imports
import multiprocessing
from multiprocessing import Process, Queue, Manager
import time

# ------------------------- CONFIG & GLOBALS -------------------------
app = Flask(__name__)
app.secret_key = 'replace_this_with_a_strong_secret'  # <-- change this

CAMERAS = {
    0: "PC Camera",
    "http://192.168.1.97:4747/video/mjpegfeed?640x480": "Phun Bijesh",  
    # "http://10.5.14.52:4747/video/mjpegfeed?640x480": "Phun Manjil", 
    # "http://10.5.21.107:4747/video/mjpegfeed?640x480": "Phun Rachana", 
}

nimgs = 10
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for directory in ['Attendance', 'static', 'static/faces', 'static/found', 'uploads']:
    os.makedirs(directory, exist_ok=True)

# removed model_path / joblib / sklearn usage
USER_DB = 'users.db'         # original users DB
ENC_DB = 'encodings.db'      # new DB for registered faces and attendance (we'll store users here too)
USER_BASEPATH = 'static/faces'

# ------------------------- SETTINGS FOR CUSTOM KNN -------------------------
K_NEIGHBORS = 3
METRIC = "cosine"  # "euclidean" or "cosine"
THRESHOLD_EUCLIDEAN = 0.5
THRESHOLD_COSINE = 0.95  # interpreted as minimum cosine similarity

# ------------------------- DATABASE SETUP -------------------------
def init_db():
    # create users db (kept) and encoding db (new combined)
    # We'll keep the original USER_DB for auth compatibility, but we can also create tables in the same DB
    # To avoid confusion, we'll create/ensure both USER_DB and ENC_DB exist and necessary tables are in ENC_DB.
    # Keep original users table in USER_DB as your app currently expects.
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

    con2 = sqlite3.connect(ENC_DB)
    cur2 = con2.cursor()
    # registered faces: owner_username ties to the logged-in user that registered these faces
    cur2.execute('''CREATE TABLE IF NOT EXISTS registered_faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        owner_username TEXT,
                        person_label TEXT,   -- e.g., "John_123" (name_id)
                        person_name TEXT,    -- "John"
                        person_id TEXT,      -- "123"
                        encoding TEXT,       -- JSON string of 128d list
                        created_at TEXT
                    )''')
    # attendance logs per owner (i.e., logged-in user)
    cur2.execute('''CREATE TABLE IF NOT EXISTS attendance_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        owner_username TEXT,
                        person_label TEXT,
                        person_name TEXT,
                        person_id TEXT,
                        time TEXT,
                        camera TEXT
                    )''')
    # search logs (per owner)
    cur2.execute('''CREATE TABLE IF NOT EXISTS search_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        owner_username TEXT,
                        search_target TEXT,
                        time TEXT,
                        status TEXT,
                        camera TEXT
                    )''')
    con2.commit()
    con2.close()

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

def get_all_registered_for_owner(owner_username):
    """Return (encodings_np_array, labels_list) for the given owner user.
       encodings_np_array shape = (N,128), labels_list length N, label as person_label (e.g., 'Name_123')"""
    con = sqlite3.connect(ENC_DB)
    cur = con.cursor()
    cur.execute("SELECT person_label, encoding FROM registered_faces WHERE owner_username = ?", (owner_username,))
    rows = cur.fetchall()
    con.close()
    labels = []
    encs = []
    for lbl, enc_json in rows:
        try:
            enc = np.array(json.loads(enc_json))
            if enc.shape[0] == 128:
                encs.append(enc)
                labels.append(lbl)
        except Exception:
            continue
    if len(encs) == 0:
        return np.array([]), []
    return np.vstack(encs), labels

def save_encoding_to_db(owner_username, person_label, person_name, person_id, encoding):
    """encoding: numpy array of length 128"""
    con = sqlite3.connect(ENC_DB)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO registered_faces (owner_username, person_label, person_name, person_id, encoding, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (owner_username, person_label, person_name, str(person_id), json.dumps(encoding.tolist()), datetime.now().isoformat())
    )
    con.commit()
    con.close()

def log_attendance_db(owner_username, person_label, person_name, person_id, camera_name):
    con = sqlite3.connect(ENC_DB)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO attendance_logs (owner_username, person_label, person_name, person_id, time, camera) VALUES (?, ?, ?, ?, ?, ?)",
        (owner_username, person_label, person_name, str(person_id), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), camera_name)
    )
    con.commit()
    con.close()

def log_search_db(owner_username, search_target, status, camera_name):
    con = sqlite3.connect(ENC_DB)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO search_logs (owner_username, search_target, time, status, camera) VALUES (?, ?, ?, ?, ?)",
        (owner_username, search_target, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, camera_name)
    )
    con.commit()
    con.close()

# ------------------------- CUSTOM KNN & METRICS -------------------------
def euclidean_distances(face_encoding, known_encodings):
    # face_encoding: (128,), known_encodings: (N,128)
    # returns distances shape (N,)
    diffs = known_encodings - face_encoding
    dists = np.linalg.norm(diffs, axis=1)
    return dists

def cosine_similarities(face_encoding, known_encodings):
    # returns cosine similarity in [-1,1] shape (N,)
    # handle zero vectors defensively
    fe = face_encoding / np.linalg.norm(face_encoding) if np.linalg.norm(face_encoding) != 0 else face_encoding
    ke_norms = np.linalg.norm(known_encodings, axis=1)
    # avoid division by zero
    valid = ke_norms != 0
    sims = np.zeros(known_encodings.shape[0])
    if known_encodings.shape[0] == 0:
        return sims
    sims[valid] = np.dot(known_encodings[valid], fe) / ke_norms[valid]
    return sims

def predict_knn_custom(face_encoding, known_encodings, known_labels, k=K_NEIGHBORS, metric=METRIC):
    """
    Returns predicted label or None if nothing within threshold.
    For euclidean: smaller distances better. Use THRESHOLD_EUCLIDEAN.
    For cosine: larger similarity better. Use THRESHOLD_COSINE.
    """
    if known_encodings.size == 0 or len(known_labels) == 0:
        return None

    if metric == "euclidean":
        dists = euclidean_distances(face_encoding, known_encodings)
        idx_sorted = np.argsort(dists)
        kidx = idx_sorted[:min(k, len(dists))]
        # threshold check: if best distance > threshold => unknown
        if dists[kidx[0]] > THRESHOLD_EUCLIDEAN:
            return None
        k_labels = [known_labels[i] for i in kidx]
        # majority vote
        pred = max(set(k_labels), key=k_labels.count)
        return pred
    elif metric == "cosine":
        sims = cosine_similarities(face_encoding, known_encodings)
        # higher is better
        idx_sorted = np.argsort(-sims)
        kidx = idx_sorted[:min(k, len(sims))]
        if sims[kidx[0]] < THRESHOLD_COSINE:
            return None
        k_labels = [known_labels[i] for i in kidx]
        pred = max(set(k_labels), key=k_labels.count)
        return pred
    else:
        # fallback to euclidean
        return predict_knn_custom(face_encoding, known_encodings, known_labels, k, "euclidean")

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
                        # some multiprocessing Queue implementations have put_bytes
                        req_queue.put_bytes(encoded.tobytes())
                    except Exception:
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
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyWindow(f'Camera - {camera_name}')
        print(f"[camera_process] {camera_name} closed")

def recognizer_process(all_req_queues, all_res_queues, shared_stop, username_for_attendance, mode_shared):
    """
    Runs in single process. Loads registered encodings for the owner (username_for_attendance)
    and performs face detection/recognition using our custom KNN/predict function.
    all_req_queues: dict camera_name -> Queue
    all_res_queues: dict camera_name -> Queue
    shared_stop: Manager().dict() with {'stop': False}
    username_for_attendance: the username to log attendance/search results for
    mode_shared: Manager().dict with {'mode': 'attendance' or 'search', 'search_user': None}
    """
    # load known encodings for this owner once at start
    known_encodings, known_labels = get_all_registered_for_owner(username_for_attendance)
    print(f"[recognizer] Loaded {len(known_labels)} registered encodings for owner {username_for_attendance}")

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
                        label = None
                        try:
                            # generate encoding for face_img (convert color order)
                            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                            else:
                                rgb_face = face_img
                            # face_recognition expects full image; use face_encodings directly
                            # resize to consistent size to mimic earlier behavior (we used fx=0.5 earlier)
                            small_face = cv2.resize(rgb_face, (0, 0), fx=0.5, fy=0.5)
                            face_locations = face_recognition.face_locations(small_face, model='hog')
                            encs = face_recognition.face_encodings(small_face, known_face_locations=face_locations)
                            if encs and len(encs) > 0:
                                enc = encs[0]
                                # Predict using custom KNN
                                label = predict_knn_custom(enc, known_encodings, known_labels, k=K_NEIGHBORS, metric=METRIC)
                        except Exception as e:
                            print(f"[recognizer] error identifying face: {e}")
                            label = None

                        detections.append((x, y, w, h, label))

                        # handle logging for attendance/search
                        if mode_shared['mode'] == 'attendance':
                            if label:
                                # label is person_label like "Name_123"
                                # we can split to name and id
                                person_name = label.split('_')[0] if '_' in label else label
                                person_id = label.split('_', 1)[1] if '_' in label else ''
                                # keep CSV file and DB logging both (maintain existing CSV behavior)
                                add_attendance(label, cam_key, username=username_for_attendance)
                                # also log in DB attendance table
                                log_attendance_db(username_for_attendance, label, person_name, person_id, cam_key)
                        
                        elif mode_shared['mode'] == 'search':
                            search_target = mode_shared.get('search_user')
                            if label == search_target:
                                # Log to file
                                search_log = user_searchlog_path(username_for_attendance)
                                with open(search_log, 'a') as f:
                                    f.write(f"{search_target},{datetime.now().strftime('%H:%M:%S')},Found,{cam_key}\n")
                                
                                # Log to database
                                log_search_db(username_for_attendance, search_target, "Found", cam_key)
                                
                                # Print confirmation
                                print(f"[recognizer] ✅ {search_target} found in {cam_key}")
                                
                                # CRITICAL: Set flags BEFORE stopping
                                shared_stop['found_user'] = search_target
                                shared_stop['found_camera'] = cam_key
                                
                                # Small delay to ensure flags are written to shared memory
                                time.sleep(0.2)
                                
                                # Now signal to stop
                                shared_stop['stop'] = True
                                break

                # return detections to camera-specific response queue
                res_q = all_res_queues.get(cam_key)
                if res_q:
                    res_q.put(detections)
            except Exception as e:
                print(f"[recognizer] error processing frame from {cam_key}: {e}")

    print("[recognizer] stopping recognizer process")
    """
    Runs in single process. Loads registered encodings for the owner (username_for_attendance)
    and performs face detection/recognition using our custom KNN/predict function.
    all_req_queues: dict camera_name -> Queue
    all_res_queues: dict camera_name -> Queue
    shared_stop: Manager().dict() with {'stop': False}
    username_for_attendance: the username to log attendance/search results for
    mode_shared: Manager().dict with {'mode': 'attendance' or 'search', 'search_user': None}
    """
    # load known encodings for this owner once at start
    known_encodings, known_labels = get_all_registered_for_owner(username_for_attendance)
    print(f"[recognizer] Loaded {len(known_labels)} registered encodings for owner {username_for_attendance}")

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
                        label = None
                        try:
                            # generate encoding for face_img (convert color order)
                            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                            else:
                                rgb_face = face_img
                            # face_recognition expects full image; use face_encodings directly
                            # resize to consistent size to mimic earlier behavior (we used fx=0.5 earlier)
                            small_face = cv2.resize(rgb_face, (0, 0), fx=0.5, fy=0.5)
                            face_locations = face_recognition.face_locations(small_face, model='hog')
                            encs = face_recognition.face_encodings(small_face, known_face_locations=face_locations)
                            if encs and len(encs) > 0:
                                enc = encs[0]
                                # Predict using custom KNN
                                label = predict_knn_custom(enc, known_encodings, known_labels, k=K_NEIGHBORS, metric=METRIC)
                        except Exception as e:
                            print(f"[recognizer] error identifying face: {e}")
                            label = None

                        detections.append((x, y, w, h, label))

                        # handle logging for attendance/search
                        if mode_shared['mode'] == 'attendance':
                            if label:
                                # label is person_label like "Name_123"
                                # we can split to name and id
                                person_name = label.split('_')[0] if '_' in label else label
                                person_id = label.split('_', 1)[1] if '_' in label else ''
                                # keep CSV file and DB logging both (maintain existing CSV behavior)
                                add_attendance(label, cam_key, username=username_for_attendance)
                                # also log in DB attendance table
                                log_attendance_db(username_for_attendance, label, person_name, person_id, cam_key)
                        elif mode_shared['mode'] == 'search':
                            search_target = mode_shared.get('search_user')
                            if label == search_target:
                                search_log = user_searchlog_path(username_for_attendance)
                                # always append a new search entry even if user already exists
                                with open(search_log, 'a') as f:
                                    f.write(f"{search_target},{datetime.now().strftime('%H:%M:%S')},Found,{cam_key}\n")
                                # DB log
                                log_search_db(username_for_attendance, search_target, "Found", cam_key)
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
        df = pd.read_csv(csv) if os.path.exists(csv) else None
        if df is not None:
            names, rolls, times = df['Name'], df['Roll'], df['Time']
            cameras, l = df.get('Camera', ['N/A'] * len(df)), len(df)
        else:
            names = rolls = times = cameras = []
            l = 0

    userlist, _, _, _ = getallusers_original()
    return render_template('home.html', names=names, rolls=rolls, times=times,
                           cameras=cameras, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, userlist=userlist, username=user)

@app.route('/start', methods=['GET'])
@login_required
def start():
    # spawn one recognizer + camera processes in attendance mode
    # Instead of checking for model_path, check if registered encodings exist for user
    owner = session['user']
    known_encodings, known_labels = get_all_registered_for_owner(owner)
    if known_encodings.size == 0:
        names, rolls, times, cameras, l = extract_attendance(session['user'])
        return render_template('home.html', names=names, rolls=rolls, times=times,
                              cameras=cameras, l=l, totalreg=totalreg(),
                              datetoday2=datetoday2,
                              mess='No registered faces found. Please add a new face first.')

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

        person_folder = f'{USER_BASEPATH}/{newusername}_{newuserid}'
        os.makedirs(person_folder, exist_ok=True)

        cap = open_camera(0)
        if cap is None:
            names, rolls, times, cameras, l = extract_attendance(session['user'])
            return render_template('home.html', names=names, rolls=rolls, times=times,
                                  cameras=cameras, l=l, totalreg=totalreg(),
                                  datetoday2=datetoday2,
                                  mess='❌ Cannot access laptop camera for registration!')

        i = 0
        saved_files = []
        print(f"\n{'='*50}")
        print("REGISTRATION MODE - Instructions:")
        print("  - Press 'C' to capture a photo")
        print("  - Press 'ESC' to cancel registration")
        print(f"  - Need {nimgs} photos total")
        print(f"{'='*50}\n")
        
        while i < nimgs:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            faces = extract_faces(frame)
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display instructions and progress
            cv2.putText(display_frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'C' to capture", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, "Press 'ESC' to cancel", (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow('Adding New User - Press C to Capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') or key == ord('C'):  # Capture photo
                if len(faces) > 0:
                    # Save the face
                    x, y, w, h = faces[0]  # Use first detected face
                    name = f'{newusername}_{i}.jpg'
                    file_path = os.path.join(person_folder, name)
                    cv2.imwrite(file_path, frame[y:y + h, x:x + w])
                    saved_files.append(file_path)
                    i += 1
                    print(f"✓ Photo {i}/{nimgs} captured")
                else:
                    print("⚠ No face detected! Please position your face in frame.")
                    cv2.putText(display_frame, "No face detected!", (30, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow('Adding New User - Press C to Capture', display_frame)
                    cv2.waitKey(1000)  # Show message for 1 second
            
            elif key == 27:  # ESC to cancel
                print("Registration cancelled by user")
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(saved_files) < nimgs:
            names, rolls, times, cameras, l = extract_attendance(session['user'])
            return render_template('home.html', names=names, rolls=rolls, times=times,
                                  cameras=cameras, l=l, totalreg=totalreg(),
                                  datetoday2=datetoday2,
                                  mess=f'⚠ Registration incomplete. Only {len(saved_files)}/{nimgs} photos captured.')

        # Process encodings
        owner = session['user']
        person_label = f"{newusername}_{newuserid}"
        encodings_saved = 0
        
        for fp in saved_files:
            try:
                img = face_recognition.load_image_file(fp)
                small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                encs = face_recognition.face_encodings(small_img, model='hog')
                if encs and len(encs) > 0:
                    enc = encs[0]
                    save_encoding_to_db(owner, person_label, newusername, newuserid, enc)
                    encodings_saved += 1
            except Exception as e:
                print(f"[add] error processing saved file {fp}: {e}")
                continue

        names, rolls, times, cameras, l = extract_attendance(session['user'])
        return render_template('home.html', names=names, rolls=rolls, times=times,
                              cameras=cameras, l=l, totalreg=totalreg(),
                              datetoday2=datetoday2,
                              mess=f'✅ User {newusername} registered successfully! ({encodings_saved} encodings saved)')

    return redirect(url_for('home'))



@app.route('/search', methods=['POST'])
@login_required
def search_user():
    user = session['user']
    searchuser = request.form['searchuser']

    # Check if there are registered faces for this owner
    owner = session['user']
    known_encodings, known_labels = get_all_registered_for_owner(owner)
    if known_encodings.size == 0:
        names, rolls, times, cameras, l = extract_attendance(session['user'])
        userlist, _, _, _ = getallusers_original()
        return render_template('home.html', names=names, rolls=rolls, times=times,
                              cameras=cameras, l=l, totalreg=totalreg(),
                              datetoday2=datetoday2, userlist=userlist,username=user,
                              mess="No registered faces for your account.", color_class='text-danger')

    # Initialize multiprocessing manager and shared flags
    manager = Manager()
    shared_stop = manager.dict()
    shared_stop['stop'] = False
    shared_stop['found_user'] = None     # Track which user was found
    shared_stop['found_camera'] = None   # Track which camera found them
    
    mode_shared = manager.dict()
    mode_shared['mode'] = 'search'
    mode_shared['search_user'] = searchuser

    # Prepare queues and processes for each camera
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

    # Start recognizer process
    recognizer = Process(target=recognizer_process, 
                        args=(all_req_queues, all_res_queues, shared_stop, 
                              session['user'], mode_shared))
    recognizer.start()

    # Wait for search to complete or timeout
    start_time = time.time()
    timeout = 300  # 5 minutes

    while not shared_stop['stop'] and time.time() - start_time < timeout:
        alive = any(p.is_alive() for p in cam_processes)
        if not alive:
            break
        time.sleep(0.1)

    # Give a moment for the found_user flag to be set before stopping everything
    if shared_stop['stop']:
        time.sleep(0.3)
    
    # Stop all processes
    shared_stop['stop'] = True
    time.sleep(0.2)

    # Cleanup camera processes
    for p in cam_processes:
        try:
            p.terminate()
            p.join(timeout=1)
        except Exception as e:
            print(f"Error terminating camera process: {e}")

    # Cleanup recognizer process
    try:
        recognizer.terminate()
        recognizer.join(timeout=1)
    except Exception as e:
        print(f"Error terminating recognizer process: {e}")

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Check if user was actually found using shared_stop flags
    found_user = shared_stop.get('found_user')
    found_camera = shared_stop.get('found_camera')
    
    # Get current attendance data
    names, rolls, times, cameras, l = extract_attendance(session['user'])
    userlist, _, _, _ = getallusers_original()

    # Generate appropriate message based on search results
    if found_user and found_camera:
        mess = f"✅ User Found: {searchuser} in {found_camera}"
        color_class = "text-success"
    elif found_user:
        mess = f"✅ User Found: {searchuser}"
        color_class = "text-success"
    else:
        mess = f"❌ User '{searchuser}' Not Found in any camera feed"
        color_class = "text-danger"

    return render_template('home.html', names=names, rolls=rolls, times=times,
                           cameras=cameras, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, userlist=userlist,
                           mess=mess, color_class=color_class)


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

# ------------------------- ATTENDANCE CSV helpers (kept) -------------------------
import pandas as pd  # used by extract_attendance / add_attendance existing logic

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

# ------------------------- MAIN -------------------------
if __name__ == '__main__':
    # necessary on Windows to avoid fork issues
    try:
        multiprocessing.set_start_method('spawn')
    except Exception:
        pass

    os.makedirs('Attendance', exist_ok=True)
    os.makedirs(USER_BASEPATH, exist_ok=True)
    # removed train_model() as encodings are created and stored at registration
    app.run(debug=True)
