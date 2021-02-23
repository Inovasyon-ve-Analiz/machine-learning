from matplotlib import pyplot as plt
import numpy as np
import cv2
import glob

def learningCurve(x, y):
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import learning_curve
    from sklearn.model_selection import ShuffleSplit
    
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, random_state=1)
    
    model = MLPClassifier(hidden_layer_sizes=3, max_iter=10000, random_state=1)
    model.fit(train_x, train_y)
    
    train_sizes = np.linspace(0.1, 1, 5)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    train_sizes, train_scores, validation_scores = learning_curve(model, x, y, cv=cv, n_jobs=-1, train_sizes=train_sizes)
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)
    
    fig, ax = plt.subplots(2, figsize=(10,15))
    ax[0].plot(train_sizes, 1 - train_scores_mean, label = 'Training Error')
    ax[0].plot(train_sizes, 1 - validation_scores_mean, label = 'Validation Error')
    ax[0].set_xlabel("Training Example"); ax[0].set_ylabel("Error")
    
    ax[1].plot(np.arange(1, model.n_iter_ + 1), model.loss_curve_, label="Training Error")
    ax[1].set_xlabel("Number of Iterations"); ax[1].set_ylabel("Training Error")
    
    ax[0].legend(); ax[1].legend(); 
    plt.subplots_adjust(hspace=0.4)
    plt.show()


a = 100
drones = np.ndarray((1,a**2))
for filename in glob.glob('../dataset/drone/*.jpg'):
    img = cv2.resize(cv2.imread(filename,0),(a,a)).reshape(1,a**2)[0]
    drones = np.vstack((drones,img))
drones = drones[1:]

antis = np.ndarray((1,a**2))
for filename in glob.glob('../dataset/anti_drone/*.jpg'):
    img = cv2.resize(cv2.imread(filename,0),(a,a)).reshape(1,a**2)[0]
    antis = np.vstack((antis,img))
antis = antis[1:]

x = np.vstack((drones,antis))
y = np.vstack((np.ones((len(drones),1)),np.zeros((len(antis),1)))).ravel()

learningCurve(x, y.ravel())




