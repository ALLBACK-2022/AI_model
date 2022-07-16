from tensorflow import keras
import numpy as np
import cv2
import runway

# Read class names
with open("class_names.txt", "r") as ins:
  class_names = []
  for line in ins:
    class_names.append(line.rstrip('\n'))

# Load the model
model = keras.models.load_model('doodleNet-model.h5')
model.summary()

# open a local image
#img = cv2.imread('apple.png')
img = cv2.imread('umbrella.png')
img = cv2.resize(img, (28, 28))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.reshape((28, 28, 1))
img = (255 - img) / 255

# predict
pred = model.predict(np.expand_dims(img, axis=0))[0]
ind = (-pred).argsort()[0:] # ind is index of classname 5
#ind = (-pred).argsort()[:5] # ind is index of classname 5
# ind = (-pred).argsort()[:10] # ind is index of classname 10
latex = [class_names[x] for x in ind] # latex is top 10 classname
print(ind[0]) # ind[0] is the highest index of classname
#print(latex) # 5개 출력됨.
ranword='umbrella'
print('----total-------')
# r= [s for s in ind if ranword in s]
# print(r)

print('----results-------')

# 정확도
# for x in range(0,len(ind)):
#   print('rank ' + str(x+1) + ': ' + latex[x])
#   print('accuarcy: ' + str(round(pred[ind[x]]*100, 2)) + '%')

results={}
for x in range(0,len(ind)):
  if(latex[x]==ranword):
    print('random의 해당 단어는'+latex[x]+'유사도는'+str(round(pred[ind[x]]*100, 2)) + '%')
  if(x<5):
    results[ latex[x] ]=str(round(pred[ind[x]]*100, 2)) + '%'
  #print('rank ' + str(x+1) + ': ' + latex[x])
  #print('accuarcy: ' + str(round(pred[ind[x]]*100, 2)) + '%')

print(results)
