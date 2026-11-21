# HomaTect
HomaTect: Automated Fault Detection &amp; Localization of Deep Learning Programs with Actionable Causal Explanations

## Project Structure and Execution Environment

Currently, HomaTect is implemented as a Jupyter Notebook and operates in conjunction with a local clone of the Keras repository. Because its implementation imports Keras modules and uses its internal functionality for runtime interception and model analysis, the HomaTect notebook must reside in the root directory of the cloned Keras repository.

The recommended directory structure is:

```text
here/
├── thenhere/                  # Cloned Keras repository
│   ├── keras
│   ├── ...
│   └── homatect.ipynb
│
└── homadebugger.ipynb         # HomaDebugger containing test cases
```

HomaDebugger, which uses HomaTect as its diagnostic core, loads the implementation using the following IPython command:

```python
%run thenhere/homatect
```

## Dependencies

HomaTect requires a local clone of the Keras source repository. It directly imports Keras modules, including internal modules under keras.src. Therefore, the Keras source version must match the version against which HomaTect was developed and tested.
The implementation was tested with Python 3.11.9, Keras 3.3.3, and TensorFlow 2.16.1.
