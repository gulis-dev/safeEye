import cv2
import torch
from flask import Flask, Response, render_template
from flask_cors import CORS
from deep_sort_realtime.deepsort_tracker import DeepSort
import mysql.connector
from datetime import datetime
import warnings
import os
import threading

# Ignorowanie ostrzeżeń FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Flask app
app = Flask(__name__)
CORS(app)

# Inicjalizacja YOLOv5
torch.multiprocessing.set_start_method('fork', force=True)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device='cpu')

# Inicjalizacja DeepSort
tracker = DeepSort(
    max_age=200,
    n_init=5,
    nn_budget=100,
    max_iou_distance=0.7
)

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)

# Licznik osób
unique_ids = set()
total_count = 0

detection_times = {}

# Lista przechowująca obiekty do rysowania
active_tracks = {}

# Konfiguracja bazy danych
db_config = {
    'host': 'localhost',
    'user': 'admin',
    'password': 'admin',
    'database': 'person_tracker'
}

# Tworzenie folderu na zapisane zdjęcia
image_folder = os.path.join(app.root_path, 'static/tracked_persons')
os.makedirs(image_folder, exist_ok=True)


# Funkcja do zapisu danych w bazie
def insert_track_data(track_id, image_path):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = "INSERT INTO tracked_persons (track_id, timestamp, image_path) VALUES (%s, %s, %s)"
        timestamp = datetime.now()
        cursor.execute(query, (track_id, timestamp, image_path))
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        conn.close()


# Funkcja do ciągłego przetwarzania kamery
def process_camera():
    global total_count, unique_ids, active_tracks, last_detection_time
    frame_counter = 0
    analysis_interval = 10

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        resized_frame = cv2.resize(frame, (640, 360))

        if frame_counter % analysis_interval == 0:
            results = yolo_model(resized_frame)
            detections = []

            for *xyxy, conf, cls in results.xyxy[0].tolist():
                if int(cls) == 0 and conf > 0.7:  # Wykrywamy tylko ludzi z pewnością > 0.7
                    x1, y1, x2, y2 = map(int, xyxy)
                    width, height = x2 - x1, y2 - y1
                    if width > 50 and height > 50:
                        frame_h, frame_w = frame.shape[:2]
                        x1 = int(x1 * frame_w / 640)
                        y1 = int(y1 * frame_h / 360)
                        x2 = int(x2 * frame_w / 640)
                        y2 = int(y2 * frame_h / 360)

                        detections.append(((x1, y1, x2, y2), conf, "person"))

            tracks = tracker.update_tracks(detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_tlbr()
                active_tracks[track_id] = bbox

                # Jeśli ID nie było wcześniej widoczne, dodaj je do słownika detection_times
                if track_id not in detection_times:
                    detection_times[track_id] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if track_id not in unique_ids:
                    unique_ids.add(track_id)
                    total_count += 1

                    # Zaktualizuj czas ostatniej detekcji
                    last_detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Zapis zdjęcia
                    detected_person = frame[:]

                    # Tworzenie unikalnej nazwy pliku
                    relative_image_path = f"tracked_persons/person_{track_id}.jpg"
                    image_path = os.path.join(image_folder, f"person_{track_id}.jpg")
                    counter = 1
                    while os.path.exists(image_path):
                        relative_image_path = f"tracked_persons/person_{track_id}_{counter}.jpg"
                        image_path = os.path.join(image_folder, f"person_{track_id}_{counter}.jpg")
                        counter += 1

                    cv2.imwrite(image_path, detected_person)

                    # Zapis danych w bazie
                    insert_track_data(track_id, relative_image_path)


# Funkcja do generowania strumienia wideo
# Dodaj zmienną przechowującą czas rozpoczęcia aplikacji
start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
last_detection_time = "N/A"

def generate_frames():
    global last_detection_time, total_count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Wyświetlanie prostokątów i ID na klatce
        for track_id, bbox in active_tracks.items():
            x1, y1, x2, y2 = map(int, bbox)

            x2 = x1 + (x2 - x1) // 2

            # Narysuj prostokąt z nową szerokością
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Dodaj czas wykrycia ID
            if track_id in detection_times:
                detection_time = detection_times[track_id]
                cv2.putText(frame, f"Detected: {detection_time}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)



        # Dodaj tło dla tekstu z informacjami
        cv2.rectangle(frame, (0, 0), (400, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Start Time: {start_time}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Last Detection: {last_detection_time}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Session Count: {total_count}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Wyświetlanie bieżącego czasu
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        frame_width = frame.shape[1]
        cv2.rectangle(frame, (frame_width - 310, 0), (frame_width, 50), (0, 0, 0), -1)
        cv2.putText(frame, current_time, (frame_width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Przetwarzanie klatki na format do transmisji
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live Stream</title>
    </head>
    <body>
        <h1>Live Video Stream</h1>
        <img src="/video_feed" style="max-width:100%; height:auto;">
    </body>
    </html>
    """


@app.route('/data')
def show_data():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = "SELECT track_id, timestamp, image_path FROM tracked_persons ORDER BY timestamp DESC"
        cursor.execute(query)
        rows = cursor.fetchall()
        html_table = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tracked Data</title>
            <style>
                table { border-collapse: collapse; width: 80%; margin: auto; }
                th, td { border: 1px solid black; padding: 8px; text-align: center; }
                th { background-color: #f2f2f2; }
                img { max-width: 100px; height: auto; }
                h1 { text-align: center; }
            </style>
        </head>
        <body>
            <h1>Tracked Data</h1>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Timestamp</th>
                    <th>Image</th>
                </tr>
        """
        for row in rows:
            track_id, timestamp, image_path = row
            html_table += f"""
                <tr>
                    <td>{track_id}</td>
                    <td>{timestamp}</td>
                    <td><img src="/static/{image_path}" alt="Person {track_id}"></td>
                </tr>
            """

        html_table += """
            </table>
        </body>
        </html>
        """
        return html_table
    except mysql.connector.Error as err:
        return f"Error: {err}"
    finally:
        cursor.close()
        conn.close()


if __name__ == '__main__':
    try:
        # Uruchomienie wątku przetwarzania kamery
        camera_thread = threading.Thread(target=process_camera, daemon=True)
        camera_thread.start()

        app.run(host='0.0.0.0', port=5761)
    finally:
        cap.release()
