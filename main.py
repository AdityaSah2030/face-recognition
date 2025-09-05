# ====================================================================
# FACE RECOGNITION ATTENDANCE SYSTEM
# ====================================================================
# This system uses face recognition to automatically mark attendance
# by comparing faces from webcam with pre-stored student images
# ====================================================================

# Import required libraries
import cv2                  # OpenCV for computer vision operations
import numpy as np         # NumPy for numerical operations and array handling
import face_recognition    # Face recognition library for encoding and matching faces
import os                 # OS module for file system operations
from datetime import datetime  # DateTime for timestamp generation

# Uncomment below line if you want to capture screen instead of webcam
# from PIL import ImageGrab

# ====================================================================
# CONFIGURATION SETTINGS
# ====================================================================

# Path to folder containing student images (roll numbers as filenames)
IMAGES_FOLDER = 'student-images'  # Change from 'Training_images' to your folder
ATTENDANCE_FILE = 'attendance.csv'  # CSV file to store attendance records

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
    Face encoding is a mathematical representation of facial features
    that can be used for comparison and recognition.
    
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
            # Generate face encoding (128-dimensional vector representing the face)
            # [0] gets the first face found in the image
            face_encoding = face_recognition.face_encodings(rgb_img)[0]
            encoding_list.append(face_encoding)
            print(f"Generated encoding for student {i+1}/{len(images)}")
        except IndexError:
            print(f"Warning: No face detected in image {i+1}. Skipping...")
            # Remove the corresponding name if no face was detected
            student_names.pop(i)
    
    return encoding_list

# ====================================================================
# STEP 3: ATTENDANCE MARKING FUNCTION
# ====================================================================

def mark_student_attendance(student_name):
    """
    Mark attendance for a recognized student in the CSV file.
    Prevents duplicate entries for the same student on the same day.
    
    Args:
        student_name: Roll number of the recognized student
    """
    try:
        # Read existing attendance records
        with open(ATTENDANCE_FILE, 'r+') as file:
            attendance_data = file.readlines()
            recorded_names = []
            
            # Extract names from existing records to check for duplicates
            for line in attendance_data:
                if line.strip():  # Skip empty lines
                    entry = line.split(',')
                    if len(entry) > 0:
                        recorded_names.append(entry[0].strip())
            
            # Only mark attendance if student hasn't been recorded today
            if student_name not in recorded_names:
                # Get current time for timestamp
                current_time = datetime.now()
                time_string = current_time.strftime('%Y-%m-%d %H:%M:%S')
                
                # Write new attendance entry
                file.write(f'{student_name},{time_string}\n')
                print(f"✓ Attendance marked for {student_name} at {time_string}")
            else:
                print(f"⚠ {student_name} already marked present today")
                
    except FileNotFoundError:
        # Create new attendance file if it doesn't exist
        with open(ATTENDANCE_FILE, 'w') as file:
            current_time = datetime.now()
            time_string = current_time.strftime('%Y-%m-%d %H:%M:%S')
            file.write(f'{student_name},{time_string}\n')
            print(f"✓ Created attendance file and marked {student_name} at {time_string}")

# ====================================================================
# OPTIONAL: SCREEN CAPTURE FUNCTION
# ====================================================================

def capture_screen_region(bbox=(300, 300, 990, 830)):
    """
    Capture a specific region of the screen instead of using webcam.
    Useful for recognizing faces in video calls or presentations.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2) defining screen region to capture
        
    Returns:
        Captured screen image in OpenCV format
    """
    # Uncomment these lines to enable screen capture:
    # screen_capture = np.array(ImageGrab.grab(bbox))
    # screen_capture = cv2.cvtColor(screen_capture, cv2.COLOR_RGB2BGR)
    # return screen_capture
    pass

# ====================================================================
# STEP 4: MAIN FACE RECOGNITION LOOP
# ====================================================================

# Generate encodings for all student images
known_face_encodings = generate_face_encodings(student_images)
print(f'✓ Face encoding generation complete for {len(known_face_encodings)} students')

# Initialize webcam - try different camera indices to find the right one
print("Searching for available cameras...")

# Try different camera indices to find your physical webcam
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
    
    # Alternative: Use screen capture instead of webcam
    # current_frame = capture_screen_region()
    
    if not success:
        print("Error: Failed to capture frame from webcam")
        break
    
    # ----------------------------------------------------------------
    # OPTIMIZE PERFORMANCE: Resize frame for faster processing
    # ----------------------------------------------------------------
    # Resize to 1/4 size for faster face detection (0.25 = 25% of original size)
    small_frame = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert from BGR (OpenCV) to RGB (face_recognition library format)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # ----------------------------------------------------------------
    # FACE DETECTION AND RECOGNITION
    # ----------------------------------------------------------------
    
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
        
        # Find the best match (minimum distance)
        best_match_index = np.argmin(face_distances)
        
        # If we found a match with good confidence
        if matches[best_match_index] and face_distances[best_match_index] < 0.6:
            # Get the student's roll number
            recognized_student = student_names[best_match_index].upper()
            
            # --------------------------------------------------------
            # DRAW BOUNDING BOX AND LABEL
            # --------------------------------------------------------
            
            # Scale back up face locations (we resized to 1/4, so multiply by 4)
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            
            # Draw green rectangle around the face
            cv2.rectangle(current_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw filled rectangle for text background
            cv2.rectangle(current_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            
            # Add student name text
            cv2.putText(current_frame, f"Roll: {recognized_student}", 
                        (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            # Mark attendance for recognized student
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
    
    # ----------------------------------------------------------------
    # DISPLAY RESULTS
    # ----------------------------------------------------------------
    
    # Show the webcam feed with recognition results
    cv2.imshow('Student Attendance System - Press Q to Quit', current_frame)
    
    # Check for quit command (press 'q' key)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print("Shutting down attendance system...")
        break

# ====================================================================
# CLEANUP
# ====================================================================

# Release webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
print("✓ Attendance system closed successfully")

# ====================================================================
# USAGE INSTRUCTIONS:
# ====================================================================
"""
1. SETUP:
    - Create 'student-images' folder in same directory as this script
    - Add student photos named with their roll numbers (e.g., '2021001.jpg')
    - Ensure attendance.csv exists (will be created automatically if not)

2. REQUIREMENTS:
    - Good lighting conditions for webcam
    - Clear, front-facing student photos
    - One face per training image

3. CONTROLS:
    - Press 'Q' to quit the application
    - Green box = Recognized student
    - Red box = Unknown person

4. ATTENDANCE FILE FORMAT:
    - CSV format: StudentRollNumber,DateTime
    - Example: 2021001,2024-09-05 14:30:25
"""