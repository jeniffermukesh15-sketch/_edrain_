Optional real data (the app auto-detects these files at start-up; delete them to go back to synthetic data):

rainfall.csv     one column named "rain" (mm/hour or mm/interval) - e.g. IMD / Kaggle / Open-Meteo history. Used to train the LSTM water-level forecaster.
drain_data.csv   columns: rain,water,waste,past,age,blocked   (blocked = 1 or 0). Used to train the blockage model.
