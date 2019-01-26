import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x, y = pkl.load(open('F:/python/samples.pkl', 'rb'))

print(x.shape, y.shape)

roll, pitch, yaw = y[:, 0], y[:, 1], y[:, 2]

print(roll.min(), roll.max(), roll.mean(), roll.std())
print(pitch.min(), pitch.max(), pitch.mean(), pitch.std())
print(yaw.min(), yaw.max(), yaw.mean(), yaw.std())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)

std = StandardScaler()
std.fit(x_train)
x_train = std.transform(x_train)
x_val = std.transform(x_val)
x_test = std.transform(x_test)

BATCH_SIZE = 64
EPOCHS = 100

model = Sequential()
model.add(Dense(units=20, activation='relu', kernel_regularizer='l2', input_dim=x.shape[1]))
model.add(Dense(units=10, activation='relu', kernel_regularizer='l2'))
model.add(Dense(units=3, activation='linear'))

print(model.summary())

callback_list = [EarlyStopping(monitor='val_loss', patience=25)]

model.compile(optimizer='adam', loss='mean_squared_error')
hist = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callback_list)
model.save('F:/python/model.h5')

print()
print('Train loss:', model.evaluate(x_train, y_train, verbose=0))
print('  Val loss:', model.evaluate(x_val, y_val, verbose=0))
print(' Test loss:', model.evaluate(x_test, y_test, verbose=0))

history = hist.history
loss_train = history['loss']
loss_val = history['val_loss']

plt.figure()
plt.plot(loss_train, label='train')
plt.plot(loss_val, label='val_loss', color='red')
plt.legend()

y_pred = model.predict(x_test)
diff = y_test - y_pred
diff_roll = diff[:, 0]
diff_pitch = diff[:, 1]
diff_yaw = diff[:, 2]

plt.figure(figsize=(16, 10))

plt.subplot(3, 1, 1)
plt.plot(diff_roll, color='red')
plt.title('roll')

plt.subplot(3, 1, 2)
plt.plot(diff_pitch, color='red')
plt.title('pitch')

plt.subplot(3, 1, 3)
plt.plot(diff_yaw, color='red')
plt.title('yaw')

plt.tight_layout()

def detect_face_points(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("C:/Users/HP/~/openface/models/dlib/shape_predictor_68_face_landmarks.dat")
    face_rect = detector(image, 1)
    if len(face_rect) != 1: return []
    
    dlib_points = predictor(image, face_rect[0])
    face_points = []
    for i in range(68):
        x, y = dlib_points.part(i).x, dlib_points.part(i).y
        face_points.append(np.array([x, y]))
    return face_points
        
def compute_features(face_points):
    #assert (len(face_points) == 68), "len(face_points) must be 68"
    face_points    
    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i+1, 68):
            features.append(np.linalg.norm(face_points[i]-face_points[j]))
            
    return np.array(features).reshape(1, -1)


#im = cv2.imread("G:\photos\pictures\1.jpg")#, cv2.IMREAD_COLOR)
#im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
##imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#face_points = detect_face_points(im)

cap = cv2.VideoCapture(0)

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, im = cap.read()
    im = cv2.resize(im, None,fx=0.3, fy=0.4, interpolation = cv2.INTER_LINEAR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    face_points = detect_face_points(im)
    for x, y in face_points:
        cv2.circle(im, (x, y), 1, (0, 255, 0), -1)
    try:
        
        features = compute_features(face_points)
        features = std.transform(features)
        model = load_model('F:/python/model.h5')
        y_pred = model.predict(features)
        
        roll_pred, pitch_pred, yaw_pred = y_pred[0]
        print(' Roll: {:.2f}°'.format(roll_pred))
        print('Pitch: {:.2f}°'.format(pitch_pred))
        print('  Yaw: {:.2f}°'.format(yaw_pred))
        print('')
    except:
        print("Look at the camera")
    
    
    
    
    
#    cv2.putText(im,'Roll',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2)    
#    cv2.putText(im,'Pitch',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2) 
#    cv2.putText(im,'Yaw',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2) 
#    
    
    plt.figure(figsize=(10, 10))
    #cv2.imshow(im)
    cv2.imshow('img',im)
    k = cv2.waitKey(30) & 0xff
    if k == 13:
        break

cap.release()
cv2.destroyAllWindows()