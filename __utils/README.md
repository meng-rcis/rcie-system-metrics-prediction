use to clean project structure

## Usage

remove all empty folders in "../models/features/source"

```bash
python ./__utils/__empty_dir.py
```

move all files in "../models/features/source" to tuning folder

```bash
python ./__utils/__move_file.py {{ metrics }}
```

join all tuning files to one file

```bash
python ./__utils/__join_file.py
```

remove col in file

```bash
python ./__utils/__remove_col.py
```

apply window sliding to file

```bash
python ./__utils/__window_size.py
```

convert file from pickle to csv

```bash
python ./__utils/__pickle_to_csv.py
```
