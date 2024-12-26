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
