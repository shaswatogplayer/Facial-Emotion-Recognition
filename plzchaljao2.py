# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import sqlite3
# import io
# import sys
#
# video = cv2.VideoCapture(0)  # Use -1 to automatically select the default camera
#
# faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# # Initialize emotion counts
# emotion_counts = {'Happy': 0, 'Sad': 0, 'Neutral': 0}
#
# # Heuristic-based emotion detection (very crude)
# def detect_emotion(face):
#     # This is a placeholder for a more complex heuristic
#     # For demonstration, let's just randomly assign emotions
#     import random
#     emotions = ['Happy', 'Sad', 'Neutral', 'Angry', 'Disgust', 'Fear', 'Surprise']
#     return random.choice(emotions)
#
# # Function to initialize the SQLite database and create the table if it does not exist
# def initialize_database():
#     conn = sqlite3.connect('userdata.db')
#     c = conn.cursor()
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS userdata (
#             name TEXT PRIMARY KEY,
#             graphfile BLOB
#         )
#     ''')
#     conn.commit()
#     conn.close()
#
# # Get the name of the user
# name = sys.argv[1]
#
# # Initialize the database and create the table if it doesn't exist
# initialize_database()
#
# try:
#     while True:
#         # Check if stop_sentinel.txt file exists
#         if os.path.exists("stop_sentinel.txt"):
#             print("Stop signal received.")
#             os.remove("stop_sentinel.txt")  # Remove the sentinel file
#             break
#
#         ret, frame = video.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = faceDetect.detectMultiScale(gray, 1.3, 5)
#         for x, y, w, h in faces:
#             sub_face_img = gray[y:y + h, x:x + w]
#             emotion = detect_emotion(sub_face_img)
#
#             # Update the count of the predicted emotion
#             emotion_counts[emotion] += 1
#
#             print("Emotion: {}".format(emotion))
#
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
#             cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
#             cv2.putText(frame, "{}".format(emotion), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#
#         cv2.imshow("Frame", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Use cv2.waitKey(1) for video feeds
#             break
#
#     # Plot the emotion counts
#     plt.figure(figsize=(10, 5))
#     colors = ['red', 'blue', 'green']
#     bars = plt.bar(list(emotion_counts.keys()), list(emotion_counts.values()), color=colors)
#     plt.title('Emotion Counts')
#     plt.ylabel('Count')
#     plt.xlabel('Emotion')
#     plt.grid(True)
#
#     for bar in bars:
#         plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
#                  f' {int(bar.get_height())}',
#                  ha='center', va='bottom')
#
#     # Save the plot to a BytesIO object
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.show()
#
#     # Convert the BytesIO object to a byte array
#     buf.seek(0)
#     graph_blob = buf.read()
#
#     # Open the SQLite database and insert the graph BLOB
#     conn = sqlite3.connect('userdata.db')
#     c = conn.cursor()
#     c.execute('''
#         INSERT OR REPLACE INTO userdata (name, graphfile) VALUES (?, ?)
#     ''', (name, sqlite3.Binary(graph_blob)))
#     conn.commit()
#     conn.close()
#
# except KeyboardInterrupt:  # Handle Ctrl+C
#     print("Interrupted")
#
# finally:  # This code runs whether the program ends normally or due to an exception
#     video.release()
#     cv2.destroyAllWindows()



import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sqlite3
import io
import sys

video = cv2.VideoCapture(0)  # Use -1 to automatically select the default camera

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize emotion counts with all possible emotions
emotion_counts = {
    'Happy': 0,
    'Sad': 0,
    'Neutral': 0,
    'Angry': 0,
    'Disgust': 0,
    'Fear': 0,
    'Surprise': 0
}

# Heuristic-based emotion detection (very crude)
def detect_emotion(face):
    # This is a placeholder for a more complex heuristic
    # For demonstration, let's just randomly assign emotions
    import random
    emotions = ['Happy', 'Sad', 'Neutral', 'Angry', 'Disgust', 'Fear', 'Surprise']
    return random.choice(emotions)

# Function to initialize the SQLite database and create the table if it does not exist
def initialize_database():
    conn = sqlite3.connect('userdata.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS userdata (
            name TEXT PRIMARY KEY,
            graphfile BLOB
        )
    ''')
    conn.commit()
    conn.close()

# Get the name of the user
name = sys.argv[1]

# Initialize the database and create the table if it doesn't exist
initialize_database()

try:
    while True:
        # Check if stop_sentinel.txt file exists
        if os.path.exists("stop_sentinel.txt"):
            print("Stop signal received.")
            os.remove("stop_sentinel.txt")  # Remove the sentinel file
            break

        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            sub_face_img = gray[y:y + h, x:x + w]
            emotion = detect_emotion(sub_face_img)

            # Update the count of the predicted emotion
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                # Handle unexpected emotions (optional)
                print(f"Unexpected emotion detected: {emotion}")

            print("Emotion: {}".format(emotion))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, "{}".format(emotion), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Use cv2.waitKey(1) for video feeds
            break

    # Plot the emotion counts
    plt.figure(figsize=(10, 5))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    bars = plt.bar(list(emotion_counts.keys()), list(emotion_counts.values()), color=colors)
    plt.title('Emotion Counts')
    plt.ylabel('Count')
    plt.xlabel('Emotion')
    plt.grid(True)

    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f' {int(bar.get_height())}',
                 ha='center', va='bottom')

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.show()

    # Convert the BytesIO object to a byte array
    buf.seek(0)
    graph_blob = buf.read()

    # Open the SQLite database and insert the graph BLOB
    conn = sqlite3.connect('userdata.db')
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO userdata (name, graphfile) VALUES (?, ?)
    ''', (name, sqlite3.Binary(graph_blob)))
    conn.commit()
    conn.close()

except KeyboardInterrupt:  # Handle Ctrl+C
    print("Interrupted")

finally:  # This code runs whether the program ends normally or due to an exception
    video.release()
    cv2.destroyAllWindows()



#chal gayaaa wait
# graph view ho sakte hai?
# haa dekhte haiwait
#wait
# ek baar phir se cam on karna
# ok
#phir se run karo
# camera capture kar raha hai?
# dekhte hai sahi se phirse
# main on kr rhi code
# suno, iss baar isne nahi kra, krta toh terminal pe values aati
# abhi dekho sirf do values hai
# Emotion: Fear
# Emotion: Happy
# phirse on kru?
#haa
# okay
# values dekhna

# ab thik hai?
#bhaut sahi se nahi aa raha
# iska graph wala section ?
# amit ke naam se tha na??
# ussi ka graph dekhte hai
# khana kha ke aati hu, aakr msg krti hu
# isko dekhna padega ki graph kaha dikhega
# thoda search kro tum otherwise ye wali app download krna
# beta.sqliteviewer.app
# jaun? 


# how to start the code :
# step 1: first see we are correct location that is inside the FullStack folder, if not run this command: cd FullStack_facialEmotionRecognition
# step 2: activate the virtual environment: run this command: .\test\Scripts\activate
# step 3: now, will start the code: streamlit run .\app_login_register.py
# step 4: remember, if the dependency file is to be changed just go to the line 81 of app_login_register.py
# step 5: for now the file written there is plzchaljao2.py
# step 6: now run the app as we run it, the database will get updated accordingly but the data will not be in human readable format.
# step 7: just go to this link and open the database userdata.db in that viewer.
# link: https://beta.sqliteviewer.app/userdata.db/table/userdata
# note: this tab is also bookmarked.
# step 8: go to the name of whose file is to be downloaded, click on the graph file and it will be downloaded in the local system.
