import os
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3
from keras.applications import VGG16

# ignore the warnings
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Read and Analyse Data

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:35:46.965891Z","iopub.execute_input":"2024-05-28T17:35:46.966385Z","iopub.status.idle":"2024-05-28T17:35:46.973473Z","shell.execute_reply.started":"2024-05-28T17:35:46.966347Z","shell.execute_reply":"2024-05-28T17:35:46.972448Z"}}
def loading_the_data(data_dir):
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)

    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            
            filepaths.append(fpath)
            labels.append(fold)

    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')

    df = pd.concat([Fseries, Lseries], axis=1)
    
    return df

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:35:50.032562Z","iopub.execute_input":"2024-05-28T17:35:50.033180Z","iopub.status.idle":"2024-05-28T17:35:50.077640Z","shell.execute_reply.started":"2024-05-28T17:35:50.033146Z","shell.execute_reply":"2024-05-28T17:35:50.076764Z"}}
data_dir = '/kaggle/input/tomatoleaf/tomato/train'
df = loading_the_data(data_dir)

df

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:35:52.249118Z","iopub.execute_input":"2024-05-28T17:35:52.249513Z","iopub.status.idle":"2024-05-28T17:35:52.259317Z","shell.execute_reply.started":"2024-05-28T17:35:52.249485Z","shell.execute_reply":"2024-05-28T17:35:52.258229Z"}}
data_balance = df.labels.value_counts()
data_balance

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:35:54.285007Z","iopub.execute_input":"2024-05-28T17:35:54.285376Z","iopub.status.idle":"2024-05-28T17:35:54.290068Z","shell.execute_reply.started":"2024-05-28T17:35:54.285347Z","shell.execute_reply":"2024-05-28T17:35:54.289180Z"}}
def custom_autopct(pct):
    total = sum(data_balance)
    val = int(round(pct*total/100.0))
    return "{:.1f}%\n({:d})".format(pct, val)


# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:35:55.989036Z","iopub.execute_input":"2024-05-28T17:35:55.989776Z","iopub.status.idle":"2024-05-28T17:35:56.205027Z","shell.execute_reply.started":"2024-05-28T17:35:55.989740Z","shell.execute_reply":"2024-05-28T17:35:56.204157Z"}}
plt.pie(data_balance, labels = data_balance.index, autopct=custom_autopct, colors = ["#FF0000", "#FF69B4", "#0000FF", "#FFFFFF", "#00FF00", "#800080", "#FFFF00", "#A52A2A", "#40E0D0", "#FFA500"])
plt.title("Data balance")
plt.axis("equal")
plt.show()

# %% [markdown]
# # Helper Functions

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:35:59.344970Z","iopub.execute_input":"2024-05-28T17:35:59.345673Z","iopub.status.idle":"2024-05-28T17:35:59.354549Z","shell.execute_reply.started":"2024-05-28T17:35:59.345640Z","shell.execute_reply":"2024-05-28T17:35:59.353425Z"}}
def model_performance(history, Epochs):
    tr_acc = history.history['accuracy']
    tr_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    
    Epochs = [i+1 for i in range(len(tr_acc))]
    
    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')
    
    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:36:02.814937Z","iopub.execute_input":"2024-05-28T17:36:02.815863Z","iopub.status.idle":"2024-05-28T17:36:02.822341Z","shell.execute_reply.started":"2024-05-28T17:36:02.815827Z","shell.execute_reply":"2024-05-28T17:36:02.821266Z"}}
def model_evaluation(model):
    train_score = model.evaluate(train_gen, verbose= 1)
    valid_score = model.evaluate(valid_gen, verbose= 1)
    test_score = model.evaluate(test_gen, verbose= 1)
    
    print("Train Loss: ", train_score[0])
    print("Train Accuracy: ", train_score[1])
    print('-' * 20)
    print("Validation Loss: ", valid_score[0])
    print("Validation Accuracy: ", valid_score[1])
    print('-' * 20)
    print("Test Loss: ", test_score[0])
    print("Test Accuracy: ", test_score[1])

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:36:05.013333Z","iopub.execute_input":"2024-05-28T17:36:05.014272Z","iopub.status.idle":"2024-05-28T17:36:05.018961Z","shell.execute_reply.started":"2024-05-28T17:36:05.014234Z","shell.execute_reply":"2024-05-28T17:36:05.017971Z"}}
def get_pred(model, test_gen):
    
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis = 1)
    
    return y_pred


# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:36:06.708633Z","iopub.execute_input":"2024-05-28T17:36:06.708985Z","iopub.status.idle":"2024-05-28T17:36:06.717521Z","shell.execute_reply.started":"2024-05-28T17:36:06.708958Z","shell.execute_reply":"2024-05-28T17:36:06.716540Z"}}
def plot_confusion_matrix(test_gen, y_pred):
    
    g_dict = test_gen.class_indices
    classes = list(g_dict.keys())
    
    cm = confusion_matrix(test_gen.classes, y_pred)

    plt.figure(figsize= (10, 10))
    plt.imshow(cm, interpolation= 'nearest', cmap= plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation= 45, fontsize=8)  
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')
    
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:36:10.342783Z","iopub.execute_input":"2024-05-28T17:36:10.343158Z","iopub.status.idle":"2024-05-28T17:36:10.349249Z","shell.execute_reply.started":"2024-05-28T17:36:10.343125Z","shell.execute_reply":"2024-05-28T17:36:10.347950Z"}}
def conv_block(filters, act='relu'):
    
    block = Sequential()
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(BatchNormalization())
    block.add(MaxPooling2D())
    
    return block

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:36:11.983852Z","iopub.execute_input":"2024-05-28T17:36:11.984543Z","iopub.status.idle":"2024-05-28T17:36:11.989729Z","shell.execute_reply.started":"2024-05-28T17:36:11.984503Z","shell.execute_reply":"2024-05-28T17:36:11.988753Z"}}
def dense_block(units, dropout_rate, act='relu'):
    
    block = Sequential()
    block.add(Dense(units, activation=act))
    block.add(BatchNormalization())
    block.add(Dropout(dropout_rate))
    
    return block

# %% [markdown]
# # Train - Test Split

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:36:14.339394Z","iopub.execute_input":"2024-05-28T17:36:14.340021Z","iopub.status.idle":"2024-05-28T17:36:14.350096Z","shell.execute_reply.started":"2024-05-28T17:36:14.339969Z","shell.execute_reply":"2024-05-28T17:36:14.349129Z"}}
train_df, ts_df = train_test_split(df, train_size = 0.8, shuffle = True, random_state = 42)

valid_df, test_df = train_test_split(ts_df, train_size = 0.5, shuffle = True, random_state = 42)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:36:16.767843Z","iopub.execute_input":"2024-05-28T17:36:16.768450Z","iopub.status.idle":"2024-05-28T17:36:29.678566Z","shell.execute_reply.started":"2024-05-28T17:36:16.768417Z","shell.execute_reply":"2024-05-28T17:36:29.677806Z"}}
batch_size = 16
img_size = (224, 224)

tr_gen = ImageDataGenerator(rescale=1. / 255)
ts_gen = ImageDataGenerator(rescale=1. / 255)


train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= False, batch_size= batch_size)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:36:36.027717Z","iopub.execute_input":"2024-05-28T17:36:36.028345Z","iopub.status.idle":"2024-05-28T17:36:42.018099Z","shell.execute_reply.started":"2024-05-28T17:36:36.028313Z","shell.execute_reply":"2024-05-28T17:36:42.016709Z"}}
g_dict = train_gen.class_indices     
classes = list(g_dict.keys())       
images, labels = next(train_gen)      

# ploting the patch size samples
plt.figure(figsize= (20, 20))

for i in range(batch_size):
    plt.subplot(4, 4, i + 1)
    image = images[i]
    plt.imshow(image)
    index = np.argmax(labels[i])  # get image index
    class_name = classes[index]   # get class of image
    plt.title(class_name, color= 'black', fontsize= 16)
    plt.axis('off')
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:36:57.339236Z","iopub.execute_input":"2024-05-28T17:36:57.340077Z","iopub.status.idle":"2024-05-28T17:36:57.345078Z","shell.execute_reply.started":"2024-05-28T17:36:57.340039Z","shell.execute_reply":"2024-05-28T17:36:57.344139Z"}}
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

class_counts = len(list(train_gen.class_indices.keys()))

# %% [markdown]
# # **T-Net Architecture Building**

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:37:04.118055Z","iopub.execute_input":"2024-05-28T17:37:04.118754Z","iopub.status.idle":"2024-05-28T17:37:04.125586Z","shell.execute_reply.started":"2024-05-28T17:37:04.118722Z","shell.execute_reply":"2024-05-28T17:37:04.124666Z"}}
def build_inner_encoder_decoder(input_shape):
    inputs = Input(shape=input_shape)
    
    # Inner encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Inner bottleneck
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Inner decoder
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Output
    outputs = Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
    
    return Model(inputs, outputs, name='inner_encoder_decoder')

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:45:32.516956Z","iopub.execute_input":"2024-05-28T17:45:32.517709Z","iopub.status.idle":"2024-05-28T17:45:32.524938Z","shell.execute_reply.started":"2024-05-28T17:45:32.517675Z","shell.execute_reply":"2024-05-28T17:45:32.524042Z"}}
def build_outer_encoder_decoder(input_shape):
    inputs = Input(shape=input_shape)
    
    # Outer encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Outer bottleneck
    inner_input_shape = (input_shape[0] // 2, input_shape[1] // 2, 64)
    inner_encoder_decoder = build_inner_encoder_decoder(inner_input_shape)
    x = inner_encoder_decoder(x)
    
    # Outer decoder
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Output
    outputs = Conv2D(input_shape[-1], (3, 3), activation='softmax', padding='same')(x)
    
    return Model(inputs, outputs, name='outer_encoder_decoder')

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:45:34.129418Z","iopub.execute_input":"2024-05-28T17:45:34.129812Z","iopub.status.idle":"2024-05-28T17:45:34.215485Z","shell.execute_reply.started":"2024-05-28T17:45:34.129782Z","shell.execute_reply":"2024-05-28T17:45:34.214639Z"}}
model = build_outer_encoder_decoder(img_shape)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()


# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:45:37.038754Z","iopub.execute_input":"2024-05-28T17:45:37.039138Z","iopub.status.idle":"2024-05-28T17:45:37.045126Z","shell.execute_reply.started":"2024-05-28T17:45:37.039107Z","shell.execute_reply":"2024-05-28T17:45:37.044312Z"}}
epochs = 20   

# %% [code] {"execution":{"iopub.status.busy":"2024-05-28T17:45:38.310535Z","iopub.execute_input":"2024-05-28T17:45:38.311304Z","iopub.status.idle":"2024-05-28T17:45:38.320180Z","shell.execute_reply.started":"2024-05-28T17:45:38.311266Z","shell.execute_reply":"2024-05-28T17:45:38.319231Z"}}
model.compile(Adamax(learning_rate= 0.001), loss= 'mean_squared_error', metrics= ['accuracy'])

# %% [code]
tnet_history = model.fit(train_gen, epochs= epochs, verbose= 1, validation_data= valid_gen, shuffle= False)

# %% [markdown]
# # Model Performance and Evaluation****

# %% [code]
model_performance(tnet_history, epochs)

# %% [code]
model_evaluation(model)

# %% [code]
y_pred = get_pred(model, test_gen)

plot_confusion_matrix(test_gen, y_pred)

# %% [markdown]
# # CNN Model Building

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T14:29:19.506394Z","iopub.execute_input":"2024-04-15T14:29:19.506782Z","iopub.status.idle":"2024-04-15T14:29:21.190008Z","shell.execute_reply.started":"2024-04-15T14:29:19.506755Z","shell.execute_reply":"2024-04-15T14:29:21.189227Z"}}
cnn_model = Sequential()

cnn_model.add(Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu", input_shape= img_shape))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D())

cnn_model.add(conv_block(32))

cnn_model.add(conv_block(64))

cnn_model.add(conv_block(128))

cnn_model.add(conv_block(256))

cnn_model.add(conv_block(512))

cnn_model.add(Flatten())

cnn_model.add(dense_block(256, 0.5))

cnn_model.add(dense_block(128, 0.3))

cnn_model.add(dense_block(64, 0.2))

cnn_model.add(dense_block(32, 0.2))

cnn_model.add(Dense(class_counts, activation = "softmax"))

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T14:29:29.379005Z","iopub.execute_input":"2024-04-15T14:29:29.379894Z","iopub.status.idle":"2024-04-15T14:29:29.420021Z","shell.execute_reply.started":"2024-04-15T14:29:29.379862Z","shell.execute_reply":"2024-04-15T14:29:29.419158Z"},"_kg_hide-input":true,"_kg_hide-output":false}
cnn_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

cnn_model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T14:30:03.131369Z","iopub.execute_input":"2024-04-15T14:30:03.131742Z","iopub.status.idle":"2024-04-15T14:41:00.881761Z","shell.execute_reply.started":"2024-04-15T14:30:03.131714Z","shell.execute_reply":"2024-04-15T14:41:00.880733Z"}}
epochs = 20   

history = cnn_model.fit(train_gen, epochs= epochs, verbose= 1, validation_data= valid_gen, shuffle= False)

# %% [markdown]
# # CNN Model Performance - Prediction

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T14:41:23.081726Z","iopub.execute_input":"2024-04-15T14:41:23.082092Z","iopub.status.idle":"2024-04-15T14:41:23.871007Z","shell.execute_reply.started":"2024-04-15T14:41:23.082065Z","shell.execute_reply":"2024-04-15T14:41:23.870052Z"}}
model_performance(history, epochs)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T14:42:02.079052Z","iopub.execute_input":"2024-04-15T14:42:02.079411Z","iopub.status.idle":"2024-04-15T14:42:38.633792Z","shell.execute_reply.started":"2024-04-15T14:42:02.079382Z","shell.execute_reply":"2024-04-15T14:42:38.632859Z"}}
model_evaluation(cnn_model)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T14:44:51.5085Z","iopub.execute_input":"2024-04-15T14:44:51.509192Z","iopub.status.idle":"2024-04-15T14:44:55.290981Z","shell.execute_reply.started":"2024-04-15T14:44:51.509159Z","shell.execute_reply":"2024-04-15T14:44:55.290052Z"}}
y_pred = get_pred(cnn_model, test_gen)

plot_confusion_matrix(test_gen, y_pred)

# %% [markdown]
# # EfficientNetB3 Model Building

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T14:45:59.762077Z","iopub.execute_input":"2024-04-15T14:45:59.762457Z","iopub.status.idle":"2024-04-15T14:46:02.414941Z","shell.execute_reply.started":"2024-04-15T14:45:59.762429Z","shell.execute_reply":"2024-04-15T14:46:02.413891Z"}}

base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape = img_shape, pooling= None)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = dense_block(128, 0.5)(x)
x = dense_block(32, 0.2)(x)
predictions = Dense(class_counts, activation = "softmax")(x)    # output layer with softmax activation


EfficientNetB3_model = Model(inputs = base_model.input, outputs = predictions)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T14:46:12.856747Z","iopub.execute_input":"2024-04-15T14:46:12.857595Z","iopub.status.idle":"2024-04-15T14:46:13.358992Z","shell.execute_reply.started":"2024-04-15T14:46:12.857561Z","shell.execute_reply":"2024-04-15T14:46:13.357989Z"}}
EfficientNetB3_model.compile(optimizer=Adamax(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

EfficientNetB3_model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T14:46:35.332944Z","iopub.execute_input":"2024-04-15T14:46:35.333319Z","iopub.status.idle":"2024-04-15T15:13:29.727127Z","shell.execute_reply.started":"2024-04-15T14:46:35.33329Z","shell.execute_reply":"2024-04-15T15:13:29.7263Z"}}
epochs = 20   

EfficientNetB3_history = EfficientNetB3_model.fit(train_gen, epochs= epochs, verbose= 1, validation_data= valid_gen, shuffle= False)

# %% [markdown]
# # EfficientNetB3 Model Performance - Prediction

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T15:14:05.638552Z","iopub.execute_input":"2024-04-15T15:14:05.639261Z","iopub.status.idle":"2024-04-15T15:14:06.331416Z","shell.execute_reply.started":"2024-04-15T15:14:05.639231Z","shell.execute_reply":"2024-04-15T15:14:06.330476Z"}}
model_performance(EfficientNetB3_history, epochs)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T15:14:09.964357Z","iopub.execute_input":"2024-04-15T15:14:09.964911Z","iopub.status.idle":"2024-04-15T15:14:40.213908Z","shell.execute_reply.started":"2024-04-15T15:14:09.964871Z","shell.execute_reply":"2024-04-15T15:14:40.212991Z"}}
model_evaluation(EfficientNetB3_model)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T15:14:44.783875Z","iopub.execute_input":"2024-04-15T15:14:44.784843Z","iopub.status.idle":"2024-04-15T15:15:05.192854Z","shell.execute_reply.started":"2024-04-15T15:14:44.784808Z","shell.execute_reply":"2024-04-15T15:15:05.19185Z"}}
y_pred = get_pred(EfficientNetB3_model, test_gen)


plot_confusion_matrix(test_gen, y_pred)

# %% [markdown]
# # VGG16 Model Building

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T15:16:16.330361Z","iopub.execute_input":"2024-04-15T15:16:16.330797Z","iopub.status.idle":"2024-04-15T15:16:16.995016Z","shell.execute_reply.started":"2024-04-15T15:16:16.330766Z","shell.execute_reply":"2024-04-15T15:16:16.994043Z"}}
base_model = VGG16(weights='imagenet', include_top=False, input_shape = img_shape, pooling= 'max')

for layer in base_model.layers[:15]:
    layer.trainable = False
    
    

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.2)(x)   # # Dropout layer to prevent overfitting
x = Dense(256, activation = 'relu')(x)
x = Dense(128, activation = 'relu')(x)
x = Dense(32, activation = 'relu')(x)
predictions = Dense(class_counts, activation = "sigmoid")(x)    # output layer with softmax activation


VGG16_model = Model(inputs = base_model.input, outputs = predictions)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T15:16:27.152954Z","iopub.execute_input":"2024-04-15T15:16:27.153399Z","iopub.status.idle":"2024-04-15T15:16:27.159771Z","shell.execute_reply.started":"2024-04-15T15:16:27.153353Z","shell.execute_reply":"2024-04-15T15:16:27.158557Z"}}
for layer in VGG16_model.layers:
    print(layer.name, layer.trainable)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T15:16:53.941847Z","iopub.execute_input":"2024-04-15T15:16:53.942452Z","iopub.status.idle":"2024-04-15T15:16:53.986384Z","shell.execute_reply.started":"2024-04-15T15:16:53.942424Z","shell.execute_reply":"2024-04-15T15:16:53.985398Z"}}
VGG16_model.compile(optimizer=Adamax(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

VGG16_model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T15:17:05.231844Z","iopub.execute_input":"2024-04-15T15:17:05.232796Z","iopub.status.idle":"2024-04-15T15:32:26.425032Z","shell.execute_reply.started":"2024-04-15T15:17:05.232764Z","shell.execute_reply":"2024-04-15T15:32:26.424167Z"}}
epochs = 20   

VGG16_history = VGG16_model.fit(train_gen, epochs= epochs, verbose= 1, validation_data= valid_gen, shuffle= False)

# %% [markdown]
# # VGG16 Model Performance - Prediction

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T15:32:35.618644Z","iopub.execute_input":"2024-04-15T15:32:35.619007Z","iopub.status.idle":"2024-04-15T15:32:36.374108Z","shell.execute_reply.started":"2024-04-15T15:32:35.61898Z","shell.execute_reply":"2024-04-15T15:32:36.373045Z"}}
model_performance(VGG16_history, epochs)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T15:32:42.352966Z","iopub.execute_input":"2024-04-15T15:32:42.35384Z","iopub.status.idle":"2024-04-15T15:33:26.388412Z","shell.execute_reply.started":"2024-04-15T15:32:42.35381Z","shell.execute_reply":"2024-04-15T15:33:26.387505Z"}}
model_evaluation(VGG16_model)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-15T15:34:33.917247Z","iopub.execute_input":"2024-04-15T15:34:33.917972Z","iopub.status.idle":"2024-04-15T15:34:40.421059Z","shell.execute_reply.started":"2024-04-15T15:34:33.917941Z","shell.execute_reply":"2024-04-15T15:34:40.419843Z"}}
y_pred = get_pred(VGG16_model, test_gen)

plot_confusion_matrix(test_gen, y_pred)
