import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import cv2
import numpy as np 

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights("./model.weights.h5")


def preprocess_for_mnist(frame):
    """
    Convert a Raspberry Pi camera frame → MNIST-like 28x28 image.
    Returns None if no valid digit is found.
    """
    if frame is None or frame.size == 0:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu thresholding for consistent results
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    # Select the largest contour assuming it's the digit
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Reject contours that are too small or too large
    if w < 10 or h < 10 or w > gray.shape[1]*0.9 or h > gray.shape[0]*0.9:
        return None

    # Crop the digit from the thresholded image
    digit = thresh[y:y+h, x:x+w]

    # Resize while keeping aspect ratio
    if w > h:
        new_w = 20
        new_h = max(1, int(20 * (h / w)))  # ensure non-zero
    else:
        new_h = 20
        new_w = max(1, int(20 * (w / h)))  # ensure non-zero

    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad the digit to 28x28
    padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

    # Normalize to [0,1] and add channel dimension
    padded = padded.astype("float32") / 255.0
    padded = np.expand_dims(padded, axis=-1)

    return padded


number1 = None
number2 = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera", frame)

    # Apply Fix #1
    processed = preprocess_for_mnist(frame)

    if processed is None:
        #qprint("No digit detected")
        continue

    img_array = np.expand_dims(processed, axis=0)

    pred = model.predict(img_array, verbose=0)
    predicted_class = int(np.argmax(pred))
    confidence = float(np.max(pred))


    cv2.putText(frame, f"{predicted_class} ({confidence:.2f})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Prediction", frame)

    print(f"Predicted: {predicted_class}, Confidence: {confidence:.3f}")

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if number1 is None:
            number1 = predicted_class
            print("Stored number1 =", number1)
        elif number2 is None:
            number2 = predicted_class
            print("Stored number2 =", number2)
        else:
            print("Both numbers already filled (press r to reset).")

    if key == ord('r'):
        number1 = None
        number2 = None
        print("Reset both numbers.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Number 1:", number1)
print("Number 2:", number2)
