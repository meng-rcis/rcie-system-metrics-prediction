# rcie-system-metrics-prediction 

## Command to run Jupyter notebook

### Windows

```bash
cd D:\nuttchai\dev\Projects\rcie\repo\rcie-system-metrics-prediction; jupyter notebook
```

### Project Structure

- To-do: Add later on

Main file located at `models/predictions/features/main.py`

- Next Action

Setup Manager 
- we create training data to train meta model 
- if prediction_steps = 3 ( training data: predict_l1n1, predict_l1n2, predict_l1n3, predict_l2n1, predict_l2n2, predict_l2n3, predict_l3n1, ... )

Main Manager 
- if prediction_steps = 3, current loop 3 ( training data: predict_l1n1, predict_l2n1, predict_l3n1, predict_l3n2, predict_l3n3 [latest] )
- if trainning dataset be like that ( training data: predict_l1n1, predict_l2n1, predict_l3n1, predict_l3n2, predict_l3n3 [latest] ), is it possible to have two dataset?
  
( training data - to train meta : predict_l1n1, predict_l1n2, predict_l1n3, predict_l2n1, predict_l2n2, predict_l2n3, predict_l3n1, ... )

( training data - to collect after meta predict - only used to display dashboard : ~~predict_l1n1, predict_l2n1,~~ predict_l3n1, predict_l3n2, predict_l3n3 [latest] )
