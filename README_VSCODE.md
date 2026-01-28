# How to Run in VS Code

## 1. Open the Project
1.  Open VS Code.
2.  Go to **File > Open Folder...**
3.  Select the folder: `/Users/aarushisinha/Desktop/college/sem8/btp/stylometry`.

## 2. Set Up Python Interpreter
1.  Open any Python file (e.g., `inference.py`).
2.  Look at the bottom-right corner of the VS Code window. You should see a Python version (e.g., `3.13.0` or `Select Interpreter`).
3.  Click it and select the entry that says **`venv`: ./venv/bin/python**.
    *   *If you don't see it:* Press `Cmd+Shift+P`, type `Python: Select Interpreter`, and choose the one pointing to `./venv/`.

## 3. Run the "Winning Model" (Training)
**Important**: Run this from the root directory so the script can find the data files.

1.  Open the Terminal (`Ctrl + ~`).
2.  Run the training script (Siamese Network):
    ```bash
    python experiments/04_pan22_siamese.py
    ```
    *This will train the model for 15 epochs and save the best one to `results_pan_siamese/`.*

## 4. Run Inference (Check new files)
To use the trained model to check if two texts are by the same author:

**Option A: Interactive Mode**
1.  Run:
    ```bash
    python inference.py --interactive
    ```
2.  Paste Text 1, hit Enter, then Paste Text 2.

**Option B: File Mode**
1.  Run:
    ```bash
    python inference.py path/to/file1.txt path/to/file2.txt
    ```

## 5. View Graphs
Go to the `results_pan_siamese` folder in the file explorer to view:
-   `roc_curve.png`
-   `confusion_matrix.png`
-   `training_curves.png`
