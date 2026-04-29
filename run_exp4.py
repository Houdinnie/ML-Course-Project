#!/usr/bin/env python3
import os; os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import sys, numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import gc

X = np.load('/tmp/Xsmall.npy')
y = np.load('/tmp/ysmall.npy')
X_tr, X_te, y_tr, y_te = train_test_split(X/255., to_categorical(y), test_size=.2, random_state=1000, stratify=to_categorical(y))
del X; gc.collect()

results = {}
for act in ['relu','leaky_relu','swish']:
    print(f'>>> {act}', flush=True)
    m = Sequential([Input((64,64,3)),Conv2D(32,(3,3),padding='same',activation=act),MaxPooling2D((2,2)),BatchNormalization(),Dropout(.3),
                    Conv2D(32,(3,3),padding='same',activation=act),MaxPooling2D((2,2)),BatchNormalization(),Dropout(.3),
                    Flatten(),Dense(256,activation=act),Dropout(.3),Dense(2,activation='softmax')])
    m.compile(optimizer=Adam(.001),loss='categorical_crossentropy',metrics=['accuracy'])
    m.fit(X_tr,y_tr,batch_size=32,epochs=5,validation_data=(X_te,y_te),verbose=2)
    loss,acc = m.evaluate(X_te,y_te,verbose=0)
    results[act] = acc
    print(f'  {act}: {acc*100:.2f}%', flush=True)
    del m; gc.collect()

print('\n=== Exp 4 Results ===', flush=True)
for k,v in results.items(): print(f'  {k}: {v*100:.2f}%', flush=True)
print('Exp 4 DONE', flush=True)