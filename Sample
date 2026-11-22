## Using HomaTect

HomaTect analyzes a Keras model to identify potential configuration and architectural faults and provides diagnostic messages with actionable recommendations. It can be used with a model that has been constructed but not compiled or trained. Calling `model.compile()` or `model.fit()` is **not a prerequisite** for model diagnosis. When these operations have not been performed, HomaTect skips the corresponding checks that depend on compilation or training information.

### Example: Multiclass Classification with a 1D CNN

The following example demonstrates HomaTect on a 1D convolutional neural network designed for multiclass classification with 28 output classes.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

# Load HomaTect
%run kerassource/HomaTect.ipynb

# Construct the model
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(72, 1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 3))
model.add(MaxPooling1D(2))
# + model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model (optional for HomaTect diagnosis)
model.compile(Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (optional for HomaTect diagnosis)
# model.fit(training_features, training_labels, batch_size=32, epochs=3, validation_split=0.1)

# Specify the user input arguments to pass to HomaTect
modeltype = "Multiclass Classification"
target = "Class Numbers"
output_shape = 28
input_shape = (72,)

# Run HomaTect
HomaTect(modeltype.casefold(), target.casefold().replace(" ", ""), input_shape, output_shape)
```

### HomaTect Parameters

HomaTect accepts four arguments:

```python
HomaTect(modeltype, target, input_shape, output_shape)
```

**1. `modeltype`**

Specifies the learning task. The accepted values are:

| Value | Description |
|---|---|
| `"Binary Classification"` | Binary classification |
| `"Multiclass Classification"` | Multiclass classification |
| `"Multilabel Classification"` | Multilabel classification |
| `"Regression"` | Regression |

The value is converted to lowercase using `casefold()` before being passed to HomaTect.

**2. `target`**

Specifies the target representation or range.

For classification tasks, the accepted values are:

| Value | Description |
|---|---|
| `"Onehot Encoding"` | Targets represented using one-hot encoding |
| `"Class Numbers"` | Targets represented as class indices/numbers |

For regression tasks, the accepted values are:

| Value | Description |
|---|---|
| `"[-1, 1]"` | Target values in the range −1 to 1 |
| `"[0, 1]"` | Target values in the range 0 to 1 |
| `"[-n, 0]"` | Target values in the range −n to 0 |
| `"[0, n]"` | Target values in the range 0 to n |
| `"[-n, n]"` | Target values in the range −n to n |

The target string is converted to lowercase and spaces are removed before it is passed to HomaTect. For example, `"Class Numbers"` becomes `"classnumbers"`.

**3. `input_shape`**

Specifies the input dimensions of the model, excluding the batch dimension and the channel dimension.

For example, for a 1D CNN whose Keras input shape is `(72, 1)`, the corresponding HomaTect argument is `(72,)`. Similarly, for a 2D CNN with a Keras input shape of `(32, 32, 3)`, the HomaTect argument is `(32, 32)`.

The batch dimension is also excluded. Therefore, the HomaTect argument represents the input dimensions without either the batch or channel dimension.

**4. `output_shape`**

Specifies the number of output classes for classification tasks or the target shape for regression tasks.

For example, a multiclass classification model with 28 output classes uses `output_shape = 28`

### Diagnostic Results

For the example above, HomaTect reports the following findings:

```text
Layer3: layer needs a non linear activation function.you can use relu.
For a multiclass classification model with 28 classes encoded as classnumbers: 
Layer 6: Activation function in the output layer is sigmoid while softmax is required
Compile parameter: Loss function is categorical_crossentropy while sparse_categorical_crossentropy is required
Layer6: For a multiclass classification model, with 28 classes, you need 28 output nodes, while there are 1 nodes
Layer5: The Dense layer receives a multi-dimensional feature map without a preceding Flatten layer. If the Dense layer is intended to operate on the complete feature vector, a Flatten layer should be added before it.
```
These messages identify potential problems and provide recommendations for addressing them. HomaTect reports diagnostic findings, it does not automatically modify or repair the model.

### Compilation and Training Are Optional

A model does not need to be compiled or trained before HomaTect can analyze it. The minimum requirement is a constructed Keras model and the appropriate task and shape parameters.

- If `model.compile()` has not been called, HomaTect skips checks that require compilation information, such as loss-function configuration.
- If `model.fit()` has not been called, HomaTect skips checks that require training information, such as validation-set configuration.

Consequently, HomaTect can be used during model development, before compilation or training, as well as on models that have already been compiled or trained.
