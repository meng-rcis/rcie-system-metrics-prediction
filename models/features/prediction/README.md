### Folder Structure

```
├── README.md          <- The top-level README for developers using this project.
│
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│

```

### Command to run the code

at the root of the project run the following command

Windows:

```bash
python .\models\features\prediction\main.py
```

MacOS:

```bash
python3 ./models/features/prediction/main.py
```

Note: Must run the main.py file from the root of the project because the predefined path is relative to the root of the project. Otherwise, you can change the path in the constant/path.py file. If not, the code will return an error.
