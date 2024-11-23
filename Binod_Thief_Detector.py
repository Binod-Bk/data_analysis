#!/usr/bin/env python
# coding: utf-8

# # Thief Detector
# ## This task tests your Image Processing skills to build a motion detection algorithm that alarms you when you have an unwanted visitor in your home.

# ## Steps
# - 1. Get the live video feed from your webcam
# - 2. Fix a scene (the place you want to monitor) and store it as a reference background image
#     - Store the first frame as the reference background frame
# - 3. For every frame, check if there is any unwanted object inside the scene you are monitoring
#     - Use **Background Subtraction** concept (**cv2.absdiff( )**)
#         - Subtract the current frame from the reference background image(frame) to see the changes in the scene
#         - If there is enormous amount of pixels distrubed in the subtraction result image
#             - unwanted visitor (place is unsafe --> alarm the authorities)
#         - If there is no enormous amount of pixels distrubed in the subtraction result image
#             - no unwanted visitor (place is safe)
# - 4. Output the text **"UNSAFE"** in **red** color on the top right of the frame when there is an intruder in the scene.
# - 5. Save the live feed
# - 6. Submit the (.ipynb) file

# ## Get live video feed from webcam [10 points]

# In[1]:


import cv2


cap = cv2.VideoCapture(0)  

print("Press 'q' to exit the video feed.")

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

   
    cv2.imshow('Live Video Feed', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


# ## Read first frame, convert to Grayscale and store it as reference background image [10 points]

# In[2]:


import cv2


cap = cv2.VideoCapture(0)  


ret, first_frame = cap.read()

if ret:
    
    reference_background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    print("Reference background image captured.")
else:
    print("Failed to capture the first frame.")


cap.release()


cv2.imshow('Reference Background', reference_background)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()


# ## Compute Absolute Difference between Current and First frame [20 points]

# In[3]:


import cv2


cap = cv2.VideoCapture(0)  


ret, first_frame = cap.read()

if ret:
    
    reference_background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    print("Reference background image captured.")
else:
    print("Failed to capture the first frame.")
    cap.release()
    exit()

print("Press 'q' to exit the video feed.")

while True:
    
    ret, current_frame = cap.read()
    
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    
    gray_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    
    abs_diff = cv2.absdiff(reference_background, gray_current_frame)
    
    
    cv2.imshow('Absolute Difference', abs_diff)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


# ## Apply threshold [5 points]

# In[4]:


# Import necessary libraries
import cv2

# Start capturing video from webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

# Read the first frame and store as reference background
ret, first_frame = cap.read()

if ret:
    # Convert the first frame to grayscale
    reference_background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    print("Reference background image captured.")
else:
    print("Failed to capture the first frame.")
    cap.release()
    exit()

print("Press 'q' to exit the video feed.")

while True:
    # Capture the current frame
    ret, current_frame = cap.read()
    
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Convert the current frame to grayscale
    gray_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between current frame and reference background
    abs_diff = cv2.absdiff(reference_background, gray_current_frame)
    
    # Apply a binary threshold to the absolute difference
    _, thresholded = cv2.threshold(abs_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Display the thresholded binary image
    cv2.imshow('Thresholded Motion', thresholded)
    
    # Exit the video feed when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# ## Find contours [10 points]

# In[5]:


# Import necessary libraries
import cv2

# Start capturing video from webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

# Read the first frame and store as reference background
ret, first_frame = cap.read()

if ret:
    # Convert the first frame to grayscale
    reference_background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    print("Reference background image captured.")
else:
    print("Failed to capture the first frame.")
    cap.release()
    exit()

print("Press 'q' to exit the video feed.")

while True:
    # Capture the current frame
    ret, current_frame = cap.read()
    
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Convert the current frame to grayscale
    gray_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between current frame and reference background
    abs_diff = cv2.absdiff(reference_background, gray_current_frame)
    
    # Apply a binary threshold to the absolute difference
    _, thresholded = cv2.threshold(abs_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded binary image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the current frame
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore small contours (noise)
            # Draw a rectangle around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the current frame with contours
    cv2.imshow('Motion Detection with Contours', current_frame)
    
    # Exit the video feed when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# ## Check if contourArea is large and draw rectangle around the object, output "UNSAFE" text in red color [30 points]

# In[6]:


# Import necessary libraries
import cv2

# Start capturing video from webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

# Read the first frame and store as reference background
ret, first_frame = cap.read()

if ret:
    # Convert the first frame to grayscale
    reference_background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    print("Reference background image captured.")
else:
    print("Failed to capture the first frame.")
    cap.release()
    exit()

print("Press 'q' to exit the video feed.")

while True:
    # Capture the current frame
    ret, current_frame = cap.read()
    
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Convert the current frame to grayscale
    gray_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between current frame and reference background
    abs_diff = cv2.absdiff(reference_background, gray_current_frame)
    
    # Apply a binary threshold to the absolute difference
    _, thresholded = cv2.threshold(abs_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded binary image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours and check for large motion
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Threshold for detecting significant motion
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw a rectangle around the contour
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle
            
            # Add the "UNSAFE" text
            cv2.putText(current_frame, "UNSAFE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display the current frame with annotations
    cv2.imshow('Motion Detection - UNSAFE Alert', current_frame)
    
    # Exit the video feed when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# ## Display images [10 points]

# In[7]:


# Import necessary libraries
import cv2

# Start capturing video from webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

# Read the first frame and store as reference background
ret, first_frame = cap.read()

if ret:
    # Convert the first frame to grayscale
    reference_background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    print("Reference background image captured.")
else:
    print("Failed to capture the first frame.")
    cap.release()
    exit()

print("Press 'q' to exit the video feed.")

while True:
    # Capture the current frame
    ret, current_frame = cap.read()
    
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Convert the current frame to grayscale
    gray_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between current frame and reference background
    abs_diff = cv2.absdiff(reference_background, gray_current_frame)
    
    # Apply a binary threshold to the absolute difference
    _, thresholded = cv2.threshold(abs_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded binary image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours and check for large motion
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Threshold for detecting significant motion
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw a rectangle around the contour
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle
            
            # Add the "UNSAFE" text
            cv2.putText(current_frame, "UNSAFE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display the reference background (grayscale)
    cv2.imshow('Reference Background (Grayscale)', reference_background)
    
    # Display the thresholded binary image
    cv2.imshow('Thresholded Binary Image', thresholded)
    
    # Display the final frame with annotations
    cv2.imshow('Motion Detection - UNSAFE Alert', current_frame)
    
    # Exit the video feed when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# ## Release objects [5 points]

# In[8]:


# Release the webcam and close all OpenCV windows
cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
print("Resources released. Exiting program.")

