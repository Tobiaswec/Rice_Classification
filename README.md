Rice Classification[¶](#Rice-Classification)
============================================

### Julian Pichler & Tobias Wecht[¶](#Julian-Pichler-&-Tobias-Wecht)

In diesem Readme ist nur eine kleiner Ausschnitt der Dokumentation des AKT Projekts. Mehr Details finden Sie im Jupyter Notebook oder in der exportierten HTML Datei.

Dataset: [https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)  
Splitted dataset (used in Notebook): [https://drive.google.com/file/d/1fFb3Ybl2\_Eq7FEYBvjThWOSdXo7wEq\_v/view?usp=sharing](https://drive.google.com/file/d/1fFb3Ybl2_Eq7FEYBvjThWOSdXo7wEq_v/view?usp=sharing)

Das Ziel unseres Projekts ist es Reis Bilder in die Klassen Arborio, Basmati, Ipsala, Jasmine, Karacadag zu klassifizieren.

Es werden dabei zwei Ansätze gewählt:

*   Selber CNN aufzubauen
*   Ein pretrained Model zu finetunen

Aufgabenstellung: Wie klein kann ein Neztwerk werden.


Conclusion[¶](#Conclusion)
==========================

Es wurde mit den Anzahl der Layers, Neuronen, und Activation Funktions(relu,elu), sowie verschieden Dropout Raten(Range:0.1-0.5), Learning Raten und Optimizer probiert.

Es wurden 10 verschieden CNNs mit absteigender Anzahl an Gewichten getestet um zu zeigen, wie wenig komplex ein CNN sein kann um trotzdem noch annehmbare ACCs zu erzielen.

Alle 10 CNNs wurden mit 5 Epochen und mit einer Batch-Size von 64 trainiert.

PRE Trained Network[¶](#PRE-Trained-Network)
--------------------------------------------

Es wurde ebenfalls versucht, bereits vortrainierte Netzwerke für die Reis-Klassifizierung einzusetzten (ImageNet). Die Netzwerke auf Basis von vortrainierten Netzwerken, lieferten jedoch extrem schlechte Ergebnisse mit einer Validation Accuracy von 20-30%. Die in den vortrainieren Netzen erkannten Muster sind um einiges komplexer als die einfachen Konturen der Reiskörner.Das ist zumindest die von uns vermutete Ursache für die schlechten Ergebnisse.

Results own CNN[¶](#Results-own-CNN)
====================================

### Runs:[¶](#Runs:)

| Type   | Training ACC | Validation ACC | Testing Balanced ACC | Anzahl Gewichte |
|--------|--------------|----------------|----------------------|-----------------|
| Run 1  | 99.24        | 99.55          | 88.087               | 1,327,013       |
| Run 2  | 98.94        | 99.06          | 97.433               | 205,548,645     |
| Run 3  | 98.56        | 99.27          | 99.047               | 205,522,917     |
| Run 4  | 97.85        | 99.31          | 99.093               | 51,381,317      |
| Run 5  | 98.25        | 99.01          | 98.72                | 25,692,069      |
| Run 6  | 97.25        | 99.38          | 98.97                | 12,845,565      |
| Run 7  | 97.02        | 99.32          | 98.51                | 4,817,270       |
| Run 8  | 96.02        | 98.17          | 97.21                | 1,605,952       |
| Run 9  | 75.19        | 92.71          | 87.34                | 401,512         |
| Run 10 | 58.50        | 87.91          | 81.74                | 150,587         |


Create own CNN Model[¶](#Create-own-CNN-Model)
----------------------------------------------

### Run 1[¶](#Run-1)

    tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(img_size[0], img_size[1], 3),
                               padding='same', activation='relu', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3),
                               padding='same', activation='relu', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(48, (3, 3),
                               padding='same', activation='relu', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(48, (3, 3),
                               padding='same', activation='relu', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3),
                               padding='same', activation='relu', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3),
                               padding='same', activation='relu', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(96, (3, 3),
                               padding='same', activation='relu', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(96, (3, 3),
                               padding='same', activation='relu', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(128, (3, 3),
                               padding='same', activation='relu', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3),
                               padding='same', activation='relu', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', use_bias=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu', use_bias=False),
        tf.keras.layers.Dropout(0.2),
    
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation='relu')
    ])

### Run 2[¶](#Run-2)

    tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(img_size[0], img_size[1], 3),
                               padding='same', activation='relu', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3),
                               padding='same', activation='relu', use_bias=False),
    
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', use_bias=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu', use_bias=False),
        tf.keras.layers.Dropout(0.2),
    
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation='relu')
    ])

Die Ergebnisse von Run 2 zeigten das Run 1 schon in das Overfitting trainiert hat, das einfachere Model von Run 2 ist genereller - daher wurde die Balanced Accuracy auf die Test Daten gesteigert

### Run 3[¶](#Run-3)

    tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(img_size[0], img_size[1], 3),
                               padding='same', activation='relu', use_bias=False),
    
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', use_bias=False),
        tf.keras.layers.Dropout(0.2),
    
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation='relu')
    ])

### Run 4[¶](#Run-4)

Es wurde von 128 auf 32 Neuronen verringert. Dabei fällt auf das es mehr Epochen braucht um auf eine ähnliche Accuarcy zu erreichen. Das Endendergebnis ist sogar noch etwas genereller und erziehlt somit einen minimal bessere test balanced accuracy.


    tf.keras.models.Sequential(\[
    tf.keras.layers.Conv2D(32, (3, 3), input\_shape\=(img\_size\[0\], img\_size\[1\], 3),
                           padding\='same', activation\='relu', use\_bias\=False),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation\='relu', use\_bias\=False),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation\='relu')
    \])
    model.compile(optimizer\='adam',
                  loss\=tf.keras.losses.SparseCategoricalCrossentropy(from\_logits\=True),
                  #'mse',
                  metrics\=\['accuracy'\]
                  )

Run 5[¶](#Run-5)
================


    tf.keras.models.Sequential(\[
    tf.keras.layers.Conv2D(16, (6, 6), input\_shape\=(img\_size\[0\], img\_size\[1\], 3),
                           padding\='same', activation\='relu', use\_bias\=False),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation\='relu', use\_bias\=False),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation\='relu')
    \])
    model.compile(optimizer\='adam',
                  loss\=tf.keras.losses.SparseCategoricalCrossentropy(from\_logits\=True),
                  #'mse',
                  metrics\=\['accuracy'\]
                  )

Run 6[¶](#Run-6)
================

* * *


    tf.keras.models.Sequential(\[
    tf.keras.layers.Conv2D(8, (3, 3), input\_shape\=(img\_size\[0\], img\_size\[1\], 3),
                           padding\='same', activation\='relu', use\_bias\=False),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation\='relu', use\_bias\=False),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation\='relu')
    \])
    model.compile(optimizer\='adam',
              loss\=tf.keras.losses.SparseCategoricalCrossentropy(from\_logits\=True),
              #'mse',
              metrics\=\['accuracy'\]
              )

Run 7[¶](#Run-7)
================


    tf.keras.models.Sequential(\[
    tf.keras.layers.Conv2D(3, (3, 3), input\_shape\=(img\_size\[0\], img\_size\[1\], 3),
                               padding\='same', activation\='relu', use\_bias\=False),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation\='relu', use\_bias\=False),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation\='relu')
    \])
    model.compile(optimizer\='adam',
                  loss\=tf.keras.losses.SparseCategoricalCrossentropy(from\_logits\=True),
                  #'mse',
                  metrics\=\['accuracy'\]
                  )

Run 8[¶](#Run-8)
================


    tf.keras.models.Sequential(\[
    tf.keras.layers.Conv2D(1, (3, 3), input\_shape\=(img\_size\[0\], img\_size\[1\], 3),
                               padding\='same', activation\='relu', use\_bias\=False),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation\='relu', use\_bias\=False),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation\='relu')
    \])
    model.compile(optimizer\='adam',
                  loss\=tf.keras.losses.SparseCategoricalCrossentropy(from\_logits\=True),
                  #'mse',
                  metrics\=\['accuracy'\]
                  )

Run 9[¶](#Run-9)
================



    tf.keras.models.Sequential(\[
    tf.keras.layers.Conv2D(1, (3, 3), input\_shape\=(img\_size\[0\], img\_size\[1\], 3),
                               padding\='same', activation\='relu', use\_bias\=False),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation\='relu', use\_bias\=False),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation\='relu')
    \])
    model.compile(optimizer\='adam',
                  loss\=tf.keras.losses.SparseCategoricalCrossentropy(from\_logits\=True),
                  #'mse',
                  metrics\=\['accuracy'\]
                  )

Run 10[¶](#Run-10)
==================


    tf.keras.models.Sequential(\[
    tf.keras.layers.Conv2D(1, (3, 3), input\_shape\=(img\_size\[0\], img\_size\[1\], 3),
                               padding\='same', activation\='relu', use\_bias\=False),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3, activation\='relu', use\_bias\=False),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation\='relu')
    \])
    model.compile(optimizer\='adam',
                  loss\=tf.keras.losses.SparseCategoricalCrossentropy(from\_logits\=True),
                  #'mse',
                  metrics\=\['accuracy'\]
                  )

### Train Model[¶](#Train-Model)

```
filepath \= base\_dir + 'rice\_{epoch:02d}\-{accuracy:.4f}.hdf5'
checkpoint \= tf.keras.callbacks.ModelCheckpoint(filepath, monitor\='accuracy', verbose\=1,
                                                save\_best\_only\=True, mode\='max')
callbacks\_list \= \[checkpoint\]

history \= model.fit(train\_generator,
                    steps\_per\_epoch\=(len(train\_generator.filenames) // batch\_size), 
                    epochs\=5,#30
                    validation\_data\=validation\_generator, 
                    validation\_steps\=(len(validation\_generator.filenames) // batch\_size),
                    callbacks\=callbacks\_list)
 ```                  

Epoch 1/5 </br>
796/796 \[==============================\] - ETA: 0s - loss: 0.2445 - accuracy: 0.9301 </br>
Epoch 1: accuracy improved from inf to 0.93007, saving model to /content/drive/MyDrive/Master/2.Semester/NDL/work/rice\_01-0.9301.hdf5 </br>
796/796 \[==============================\] - 654s 810ms/step - loss: 0.2445 - accuracy: 0.9301 - val\_loss: 0.0395 - val\_accuracy: 0.9887 </br>
Epoch 2/5 </br>
796/796 \[==============================\] - ETA: 0s - loss: 0.0894 - accuracy: 0.9731 </br>
Epoch 2: accuracy did not improve from 0.93007 </br>
796/796 \[==============================\] - 643s 807ms/step - loss: 0.0894 - accuracy: 0.9731 - val\_loss: 0.0249 - val\_accuracy: 0.9932 </br>
Epoch 3/5 </br>
796/796 \[==============================\] - ETA: 0s - loss: 0.0810 - accuracy: 0.9742 </br>
Epoch 3: accuracy did not improve from 0.93007 </br>
796/796 \[==============================\] - 638s 802ms/step - loss: 0.0810 - accuracy: 0.9742 - val\_loss: 0.0219 - val\_accuracy: 0.9937 </br>
Epoch 4/5 </br>
796/796 \[==============================\] - ETA: 0s - loss: 0.0688 - accuracy: 0.9779 </br>
Epoch 4: accuracy did not improve from 0.93007 </br>
796/796 \[==============================\] - 650s 816ms/step - loss: 0.0688 - accuracy: 0.9779 - val\_loss: 0.0239 - val\_accuracy: 0.9943 </br>
Epoch 5/5 </br>
796/796 \[==============================\] - ETA: 0s - loss: 0.0620 - accuracy: 0.9801 </br>
Epoch 5: accuracy did not improve from 0.93007 </br>
796/796 \[==============================\] - 674s 847ms/step - loss: 0.0620 - accuracy: 0.9801 - val\_loss: 0.0342 - val\_accuracy: 0.9887 </br>


# Evaluation

Class Arborio: </br>
    Sensitivity (TPR):  95.067% (2852 of 3000) </br>
    Specificity (TNR):  99.767% (11972 of 12000) </br>
    Precision:          99.028% (2852 of 2880) </br>
    Neg. pred. value:   98.779% (11972 of 12120) </br>
    
Class Basmati: </br>
    Sensitivity (TPR):  99.200% (2976 of 3000) </br>
    Specificity (TNR):  99.750% (11970 of 12000) </br>
    Precision:          99.002% (2976 of 3006) </br>
    Neg. pred. value:   99.800% (11970 of 11994) </br>
    
Class Ipsala:
    Sensitivity (TPR):  99.833% (2995 of 3000) </br>
    Specificity (TNR): 100.000% (12000 of 12000) </br>
    Precision:         100.000% (2995 of 2995) </br>
    Neg. pred. value:   99.958% (12000 of 12005) </br>
    
Class Jasmine: </br>
    Sensitivity (TPR):  98.767% (2963 of 3000) </br>
    Specificity (TNR):  99.700% (11964 of 12000) </br>
    Precision:          98.800% (2963 of 2999) </br>
    Neg. pred. value:   99.692% (11964 of 12001) </br>
    
Class Karacadag: </br>
    Sensitivity (TPR):  99.433% (2983 of 3000) </br>
    Specificity (TNR):  98.858% (11863 of 12000) </br>
    Precision:          95.609% (2983 of 3120) </br>
    Neg. pred. value:   99.857% (11863 of 11880) </br>

Overall accuracy:   98.460% (14769 of 15000) </br>
Balanced accuracy:  98.460% </br>

