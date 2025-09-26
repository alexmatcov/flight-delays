# flight-delays

The goal of this project is to build a data pipeline to analyze flight delays
in relation to weather patterns.

## Getting the Data

To fetch the flight delay data, run the following:

```shell
mkdir data
curl -L -o data/flight-delay-dataset-2018-2024.zip\
  https://www.kaggle.com/api/v1/datasets/download/shubhamsingh42/flight-delay-dataset-2018-2024
unzip data/flight-delay-dataset-2018-2024.zip -d data
```

To fetch the weather data (hourly weather patterns for each airport location), run:

```shell
uv run fetch-weather.py
```

This can take a long time to execute.

## Data Structure

| Dataset       | Structure docs                                                                                                 |
| ------------- | -------------------------------------------------------------------------------------------------------------- |
| Flight Delays | [TranStats data on Kaggle](https://www.kaggle.com/datasets/shubhamsingh42/flight-delay-dataset-2018-2024/data) |
| Weather       | [Meteostat Docs](https://dev.meteostat.net/python/hourly.html#data-structure)                                  |
