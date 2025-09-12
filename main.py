# ====================================================================
# FACE RECOGNITION ATTENDANCE SYSTEM
# ====================================================================
# This system uses face recognition to automatically mark attendance
# by comparing faces from webcam with pre-stored student images
# Saves attendance data in JSON format
# ====================================================================

# Import required libraries
import cv2                  # OpenCV for computer vision operations
import numpy as np         # NumPy for numerical operations and array handling
import face_recognition    # Face recognition library for encoding and matching faces
import os                 # OS module for file system operations
import json               # JSON module for handling JSON data
from datetime import datetime  # DateTime for timestamp generation

# ====================================================================
# CONFIGURATION SETTINGS
# ====================================================================

# Path to folder containing student images (roll numbers as filenames)
IMAGES_FOLDER = 'student-images'  # Folder with student photos
ATTENDANCE_FILE = 'attendance.json'  # JSON file to store attendance records

# ====================================================================
# STEP 1: LOAD STUDENT IMAGES AND EXTRACT NAMES
# ====================================================================

print("Loading student images...")

# Initialize lists to store images and student names (roll numbers)
student_images = []     # Will store the actual image data
student_names = []      # Will store roll numbers (extracted from filenames)

# Get list of all files in the student images directory
image_files = os.listdir(IMAGES_FOLDER)
print(f"Found {len(image_files)} image files: {image_files}")

# Process each image file in the directory
for filename in image_files:
    # Read the image using OpenCV
    current_image = cv2.imread(f'{IMAGES_FOLDER}/{filename}')
    
    # Check if image was loaded successfully
    if current_image is not None:
        student_images.append(current_image)
        # Extract roll number by removing file extension (.jpg, .png, etc.)
        roll_number = os.path.splitext(filename)[0]
        student_names.append(roll_number)
        print(f"Loaded image for student: {roll_number}")
    else:
        print(f"Warning: Could not load image {filename}")

print(f"Successfully loaded {len(student_names)} student images")
print(f"Student roll numbers: {student_names}")

# ====================================================================
# STEP 2: GENERATE FACE ENCODINGS FOR ALL STUDENTS
# ====================================================================

def generate_face_encodings(images):
    """
    Generate face encodings for all student images.
    
    Args:
        images: List of student images
        
    Returns:
        List of face encodings (128-dimensional vectors)
    """
    print("Generating face encodings for student images...")
    encoding_list = []
    
    for i, img in enumerate(images):
        # Convert image from BGR (OpenCV format) to RGB (face_recognition format)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            # Generate face encoding
            face_encoding = face_recognition.face_encodings(rgb_img)[0]
            encoding_list.append(face_encoding)
            print(f"Generated encoding for student {i+1}/{len(images)}")
        except IndexError:
            print(f"Warning: No face detected in image {i+1}. Skipping...")
            # Remove the corresponding name if no face was detected
            if i < len(student_names):
                student_names.pop(i)
    
    return encoding_list

# ====================================================================
# STEP 3: JSON ATTENDANCE MARKING FUNCTION
# ====================================================================

def mark_student_attendance(student_roll_no):
    """
    Mark attendance for a recognized student in JSON format.
    Prevents duplicate entries for the same student on the same day.
    
    Args:
        student_roll_no: Roll number of the recognized student
    """
    try:
        # Initialize default JSON structure
        attendance_data = {"recognizedStudents": []}
        
        # Load existing attendance data if file exists
        if os.path.exists(ATTENDANCE_FILE):
            try:
                with open(ATTENDANCE_FILE, 'r') as file:
                    content = file.read().strip()
                    if content:  # Only parse if file has content
                        attendance_data = json.loads(content)
                        # Ensure proper structure
                        if "recognizedStudents" not in attendance_data:
                            attendance_data["recognizedStudents"] = []
                    else:
                        attendance_data = {"recognizedStudents": []}
            except (json.JSONDecodeError, FileNotFoundError):
                print("Warning: Invalid JSON file. Creating new attendance record.")
                attendance_data = {"recognizedStudents": []}
        
        # Get current date for duplicate checking
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check if student already marked present today
        already_present = False
        for record in attendance_data["recognizedStudents"]:
            if record.get("rollNo") == student_roll_no:
                # Parse existing timestamp to check if it's from today
                try:
                    record_date = record.get("timestamp", "")[:10]  # Get YYYY-MM-DD part
                    if record_date == current_date:
                        already_present = True
                        break
                except:
                    continue
        
        if not already_present:
            # Create ISO 8601 timestamp
            current_time = datetime.now()
            iso_timestamp = current_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Create new attendance record
            new_record = {
                "rollNo": student_roll_no,
                "timestamp": iso_timestamp
            }
            
            # Add to attendance data
            attendance_data["recognizedStudents"].append(new_record)
            
            # Save to JSON file
            with open(ATTENDANCE_FILE, 'w') as file:
                json.dump(attendance_data, file, indent=2)
            
            print(f"✓ Attendance marked for {student_roll_no} at {iso_timestamp}")
        else:
            print(f"⚠ {student_roll_no} already marked present today")
            
    except Exception as e:
        print(f"Error marking attendance: {e}")
        # Create fresh attendance file
        current_time = datetime.now()
        iso_timestamp = current_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        fresh_data = {
            "recognizedStudents": [
                {
                    "rollNo": student_roll_no,
                    "timestamp": iso_timestamp
                }
            ]
        }
        
        try:
            with open(ATTENDANCE_FILE, 'w') as file:
                json.dump(fresh_data, file, indent=2)
            print(f"✓ Created new attendance.json and marked {student_roll_no}")
        except Exception as write_error:
            print(f"Failed to create attendance file: {write_error}")

# ====================================================================
# STEP 4: MAIN FACE RECOGNITION LOOP
# ====================================================================

# Generate encodings for all student images
known_face_encodings = generate_face_encodings(student_images)
print(f'✓ Face encoding complete for {len(known_face_encodings)} students')

# Initialize webcam - automatically detect available cameras
print("Searching for available cameras...")

camera_index = None
for i in range(1,5):  # Check cameras 0-4
    print(f"Trying camera index {i}...")
    test_cap = cv2.VideoCapture(i)
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret and frame is not None:
            print(f"✓ Found working camera at index {i}")
            camera_index = i
            test_cap.release()
            break
        test_cap.release()
    else:
        print(f"✗ Camera index {i} not available")

if camera_index is None:
    print("Error: No working camera found!")
    exit()

print(f"Using camera index {camera_index}")
video_capture = cv2.VideoCapture(camera_index)

# Check if webcam opened successfully
if not video_capture.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Face recognition system is now running!")
print("Press 'q' to quit the application")

# Main recognition loop
while True:
    # Capture frame from webcam
    success, current_frame = video_capture.read()
    
    if not success:
        print("Error: Failed to capture frame from webcam")
        break
    
    # Resize frame for faster processing (1/4 size)
    small_frame = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert from BGR to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    # Generate encodings for all detected faces
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    # Process each detected face
    for face_encoding, face_location in zip(face_encodings, face_locations):
        
        # Compare current face with all known student faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        # Calculate face distance (lower = better match)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        # Find the best match
        best_match_index = np.argmin(face_distances)
        
        # If we found a good match
        if matches[best_match_index] and face_distances[best_match_index] < 0.6:
            # Get the student's roll number
            recognized_student = student_names[best_match_index].upper()
            
            # Scale back face locations (multiply by 4)
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            
            # Draw green rectangle around recognized face
            cv2.rectangle(current_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(current_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(current_frame, f"Roll: {recognized_student}", 
                        (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            # Mark attendance in JSON
            mark_student_attendance(recognized_student)
        
        else:
            # Unknown person detected
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            
            # Draw red rectangle for unknown person
            cv2.rectangle(current_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(current_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(current_frame, "UNKNOWN", 
                        (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    # Display the webcam feed
    cv2.imshow('Student Attendance System - Press Q to Quit', current_frame)
    
    # Check for quit command
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print("Shutting down attendance system...")
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
print("✓ Attendance system closed successfully")

# ====================================================================
# FINAL JSON OUTPUT EXAMPLE
# ====================================================================
"""
The attendance.json file will contain:
{
    "recognizedStudents": [
        {
            "rollNo": "151751515151515",
            "timestamp": "2025-09-05T15:30:10Z"
        },
        {
            "rollNo": "151751515151516",
            "timestamp": "2025-09-05T15:30:12Z"
        },
        {
            "rollNo": "151751515151517",
            "timestamp": "2025-09-05T15:30:15Z"
        }
    ]
}
"""