from flask import Flask, render_template, request, redirect, url_for, Response
from flask_sqlalchemy import SQLAlchemy
import face_recognition
import numpy as np
import os
import cv2
from datetime import datetime

# Flask App and MySQL Configuration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@localhost/flask_attendance'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Student Model
class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    image_file = db.Column(db.String(100), nullable=False)
    embedding = db.Column(db.PickleType, nullable=False)

# Attendance Model
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)  # Use DateTime instead of String

# Create the Database and Tables
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        file = request.files['file']
        if name and file:
            filename = f"{name}.jpg"
            filepath = os.path.join('./static/uploads', filename)
            file.save(filepath)

            # Load the image file and get the face encoding
            image = face_recognition.load_image_file(filepath)
            face_encoding = face_recognition.face_encodings(image)[0]

            # Add new student to the database with embedding
            new_student = Student(name=name, image_file=filename, embedding=face_encoding)
            db.session.add(new_student)
            db.session.commit()
            return redirect(url_for('home'))
    return render_template('add_student.html')

def gen_frames():
    """Capture frames from the webcam, detect faces, and display rectangles."""
    camera = cv2.VideoCapture(0)  # Open webcam
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize frame for faster processing (optional)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]  # Convert from BGR to RGB

            # Find all face locations in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)

            # Loop over face locations and draw rectangles around them
            for (top, right, bottom, left) in face_locations:
                # Scale back up the face locations since we resized the frame
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw the rectangle around each face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Encode the frame with rectangles and yield it
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route to capture video"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_attendance', methods=['GET', 'POST'])
def take_attendance():
    if request.method == 'POST':
        # Capture a frame from the webcam
        camera = cv2.VideoCapture(0)
        success, frame = camera.read()
        if success:
            filepath = './static/uploads/temp.jpg'
            cv2.imwrite(filepath, frame)
            camera.release()

            # Load the uploaded image and get its face encodings
            image = face_recognition.load_image_file(filepath)
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) > 0:
                # Retrieve student embeddings from the database
                students = Student.query.all()
                student_encodings = [np.array(s.embedding) for s in students]

                # Threshold for recognizing a face (lower values = stricter matching)
                threshold = 0.6

                # Compare faces and record attendance
                for face_encoding in face_encodings:
                    face_distances = face_recognition.face_distance(student_encodings, face_encoding)
                    
                    # Find the best match (smallest distance)
                    best_match_index = np.argmin(face_distances)
                    best_match_distance = face_distances[best_match_index]

                    if best_match_distance < threshold:
                        # Face recognized
                        student_id = students[best_match_index].id
                        new_attendance = Attendance(
                            student_id=student_id,
                            timestamp=datetime.now()  # Use current system time
                        )
                        db.session.add(new_attendance)
                        db.session.commit()
                    else:
                        # Face is unknown
                        print("Unknown face detected")

        return redirect(url_for('home'))
    return render_template('take_attendance.html')

@app.route('/view_attendance')
def view_attendance():
    # Query to fetch attendance records and corresponding student information
    attendance_records = db.session.query(Attendance, Student).join(Student).all()

    # Format timestamp before passing it to the template
    formatted_records = []
    for attendance, student in attendance_records:
        # Format the timestamp as YYYY-MM-DD HH:MM:SS
        formatted_timestamp = attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        formatted_records.append((attendance, student, formatted_timestamp))

    return render_template('view_attendance.html', attendance_records=formatted_records)

@app.route('/view_students')
def view_students():
    # Query all students from the database
    students = Student.query.all()
    return render_template('view_students.html', students=students)


if __name__ == '__main__':
    app.run(debug=True)
