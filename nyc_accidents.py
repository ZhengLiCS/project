from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

import utils


class MatplotlibTimeSeriesVisualization(utils.MatplotlibTimeSeriesVisualization):

    @staticmethod
    def time_query(dataset, date_attr, group_attr, attributes):
        date_categories = sorted(list(dataset[date_attr].value_counts().index.values))

        segments = []
        for attr in attributes:
            fact_table = dataset[[date_attr, group_attr, attr]].dropna()
            for cat in fact_table[group_attr].value_counts().index.values:
                series = fact_table[fact_table[group_attr] == cat]
                x = np.array([date_categories.index(date) for date in series[date_attr].values])
                y = series[attr].values.flatten()
                y = np.convolve(y, [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05], mode="same")  # 7-days average
                boundary = [np.min(x), np.max(x), np.min(y), np.max(y)]
                segments.append({"category": cat, "attribute": attr, "x": x, "y": y, "boundary": boundary})

        return segments, attributes, date_categories

    @classmethod
    def unit_test(cls):
        dataset = Preprocessing(pd.read_csv("src/NYC Accidents 2020.csv")).dataset

        time_series_attributes = [
            "NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED",
            "NUMBER OF PEDESTRIANS INJURED", "NUMBER OF PEDESTRIANS KILLED",
            "NUMBER OF CYCLIST INJURED", "NUMBER OF CYCLIST KILLED",
            "NUMBER OF MOTORIST INJURED", "NUMBER OF MOTORIST KILLED"
        ]

        cls(
            cache_file="cache/nyc-accidents-time-series.pkl",
            dataset=dataset,
            date_attr="CRASH DATE",
            group_attr="BOROUGH",
            attributes=time_series_attributes
        )


class MatplotlibTimeFreeVisualization(utils.MatplotlibTimeFreeVisualization):
    @classmethod
    def unit_test(cls):
        dataset = Preprocessing(pd.read_csv("src/NYC Accidents 2020.csv")).dataset
        one_hot = pd.get_dummies(dataset["BOROUGH"])
        dataset = dataset[["CRASH DATE"]].join(one_hot)
        cls(
            cache_file="cache/nyc-accidents-time-free.pkl",
            dataset=dataset,
            group_attr="CRASH DATE",
            attributes=list(one_hot.columns)
        )


class Preprocessing(utils.Preprocessing):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

        self.dataset = df
        self.dataset.info()  # TODO: remove this line

        self.dataset = self.data_integration(self.dataset)
        self.dataset = self.drop_missing_data(self.dataset)
        self.dataset = self.fill_missing_data(self.dataset)
        self.dataset.info()  # TODO: remove this line

    def __call__(self, samples, *args, **kwargs):
        samples = self.data_integration(samples)
        samples = self.drop_missing_data(samples)
        return self.fill_missing_data(samples)

    def data_integration(self, samples):
        attributes = [
            "CRASH DATE", "CRASH TIME",
            "BOROUGH",
            "LATITUDE", "LONGITUDE",
            "ON STREET NAME", "CROSS STREET NAME", "OFF STREET NAME",
            "NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED", "NUMBER OF PEDESTRIANS INJURED", "NUMBER OF PEDESTRIANS KILLED", "NUMBER OF CYCLIST INJURED", "NUMBER OF CYCLIST KILLED", "NUMBER OF MOTORIST INJURED", "NUMBER OF MOTORIST KILLED",
        ]
        fact_table = samples[attributes]

        attributes = [
            "CONTRIBUTING FACTOR VEHICLE 1", "CONTRIBUTING FACTOR VEHICLE 2", "CONTRIBUTING FACTOR VEHICLE 3",
            "CONTRIBUTING FACTOR VEHICLE 4", "CONTRIBUTING FACTOR VEHICLE 5",
        ]
        series = samples[attributes].fillna("NULL")
        values = [set(items) - {"NULL"} for items in series.values]
        fact_table.insert(fact_table.columns.__len__(), "CONTRIBUTING FACTOR VEHICLE", values)
        invalid_indices = [i for i, items in enumerate(values) if items == set()]

        attributes = [
            "VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3", "VEHICLE TYPE CODE 4",
            "VEHICLE TYPE CODE 5"
        ]
        series = samples[attributes].fillna("NULL")
        values = [set(items) - {"NULL"} for items in series.values]
        fact_table.insert(fact_table.columns.__len__(), "VEHICLE TYPE CODE", values)
        invalid_indices += [i for i, items in enumerate(values) if items == set()]

        return fact_table.drop(index=invalid_indices)

    def drop_missing_data(self, samples):
        """ All examples should have fully value in attributes `CRASH TIME`, `LATITUDE` and `LONGITUDE`.
        """
        special_data = samples[["CRASH TIME", "LATITUDE", "LONGITUDE"]]
        invalid_rows = special_data[special_data.isnull().any(axis=1)]
        return samples.drop(invalid_rows.index)

    def fill_missing_data(self, samples):
        return samples.fillna("NULL")

    @classmethod
    def unit_test(cls):
        cls(pd.read_csv("src/NYC Accidents 2020.csv"))


class Mining:
    def __init__(self, df: pd.DataFrame):
        self.dataset = df

    def clustering(self, method):
        """
        :param method: K-means, DBSCAN, etc
        :return:
        """
        pass

    def regression(self):
        # convert date to day of week
        dataset = pd.to_datetime(self.dataset["CRASH DATE"]).dt.day_name()
        dataset = pd.get_dummies(dataset)
        # binning to hours
        hour_series = {"CRASH TIME": [int(time[:2]) for time in self.dataset["CRASH TIME"].values]}
        hour_series = pd.DataFrame(hour_series, index=dataset.index)
        dataset = dataset.join(hour_series)
        # add the location message
        dataset = dataset.join(self.dataset[["LATITUDE", "LONGITUDE"]])

        features = np.asarray(dataset.values, dtype=np.float32)
        label_attributes = [
            "NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED",
            "NUMBER OF PEDESTRIANS INJURED", "NUMBER OF PEDESTRIANS KILLED",
            "NUMBER OF CYCLIST INJURED", "NUMBER OF CYCLIST KILLED",
            "NUMBER OF MOTORIST INJURED", "NUMBER OF MOTORIST KILLED"
        ]
        labels = np.asarray(self.dataset[label_attributes].values, dtype=np.float32)

        random_indices = np.arange(features.shape[0])
        np.random.shuffle(random_indices)
        x_train, y_train = features[random_indices[:60000], :], labels[random_indices[:60000], :]
        x_test, y_test = features[random_indices[60000:], :], labels[random_indices[60000:], :]

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(label_attributes.__len__())
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.005),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['accuracy'])
        model.fit(
            x_train, y_train, batch_size=32, epochs=10,
            validation_data=(x_test, y_test),
            callbacks=[
                tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: 0.75 ** (epoch // 10) * lr, verbose=1),
                tf.keras.callbacks.TensorBoard(log_dir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
                                               histogram_freq=1)
            ],
        )
        # tensorboard --logdir logs

    def association_mining(self):
        examples = [list(item_set) for item_set in self.dataset["CONTRIBUTING FACTOR VEHICLE"]]
        utils.AssociationMining().fit(
            data=examples,
            min_support=0.01,
            min_confidence=0.6
        )
        examples = [list(item_set) for item_set in self.dataset["VEHICLE TYPE CODE"]]
        utils.AssociationMining().fit(
            data=examples,
            min_support=0.01,
            min_confidence=0.4
        )

    @classmethod
    def unit_test(cls):
        print("=" * 16 + " data preprocessing " + "=" * 16)
        dataset = Preprocessing(pd.read_csv("src/NYC Accidents 2020.csv")).dataset

        print("=" * 16 + " regression " + "=" * 16)
        cls(dataset).regression()

        print("=" * 16 + " association mining " + "=" * 16)
        cls(dataset).association_mining()


if __name__ == "__main__":
    # MatplotlibTimeSeriesVisualization.unit_test()
    # MatplotlibTimeFreeVisualization.unit_test()

    # Preprocessing.unit_test()
    Mining.unit_test()
