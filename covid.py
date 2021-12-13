import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import folium

import utils

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


class MatplotlibTimeSeriesVisualization(utils.MatplotlibTimeSeriesVisualization):
    @classmethod
    def unit_test(cls):
        iso_code = list(pd.read_csv("src/countries_codes_and_coordinates.csv")["Alpha-3 code"].values)
        iso_code = [code.replace(' ', '').replace('"', '') for code in iso_code]
        dataset = pd.read_csv("src/owid-covid-data.csv")
        dataset = dataset.query("iso_code == @iso_code")

        time_series_attributes = [
            "total_cases", "total_deaths",
            "reproduction_rate", "icu_patients", "hosp_patients", "total_tests", "positive_rate",
            "total_vaccinations", "people_vaccinated", "people_fully_vaccinated", "stringency_index"
        ]

        cls(
            cache_file="cache/covid-19-time-series.pkl",
            dataset=dataset,
            date_attr="date",
            group_attr="location",
            attributes=time_series_attributes
        )


class MatplotlibTimeFreeVisualization(utils.MatplotlibTimeFreeVisualization):
    @classmethod
    def unit_test(cls):
        iso_code = list(pd.read_csv("src/countries_codes_and_coordinates.csv")["Alpha-3 code"].values)
        iso_code = [code.replace(' ', '').replace('"', '') for code in iso_code]
        dataset = pd.read_csv("src/owid-covid-data.csv")
        dataset = dataset.query("iso_code == @iso_code")

        time_free_attributes = [
            "population", "population_density", "median_age", "aged_65_older",
            "aged_70_older", "gdp_per_capita", "extreme_poverty", "cardiovasc_death_rate",
            "diabetes_prevalence", "female_smokers", "male_smokers", "handwashing_facilities",
            "hospital_beds_per_thousand", "life_expectancy", "human_development_index"
        ]

        dataset.info()

        cls(
            cache_file="cache/covid-19-time-free.pkl",
            dataset=dataset,
            group_attr="location",
            attributes=time_free_attributes
        )


class QtTimeSeriesVisualization(utils.TimeSeriesVisualization):
    pass


class JSTimeSeriesVisualization(utils.TimeSeriesVisualization):
    pass


class MapVisualization(utils.TimeFreeVisualization):
    @classmethod
    def unit_test(cls):
        locations = [item["properties"]["ADMIN"] for item in pd.read_json("src/world-locations.json")["features"]]
        iso_codes = [item["properties"]["ISO_A3"] for item in pd.read_json("src/world-locations.json")["features"]]

        dataset = Preprocessing(pd.read_csv("src/owid-covid-data.csv")).dataset
        dataset = [[locations[iso_codes.index(code)], seq[-1]] for code, seq in zip(dataset.index, dataset["total_cases"].values)]
        dataset = pd.DataFrame(data=dataset, columns=["location", "value"])

        m = folium.Map(location=[0, 0], zoom_start=3)
        folium.Choropleth(
            geo_data="src/world-countries.json",
            name="choropleth",
            data=dataset,
            columns=["location", "value"],
            key_on="feature.properties.name",
            fill_color="OrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="total_cases",
        ).add_to(m)
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('Stamen Toner').add_to(m)
        folium.TileLayer('Stamen Water Color').add_to(m)
        folium.TileLayer('cartodbpositron').add_to(m)
        folium.TileLayer('cartodbdark_matter').add_to(m)
        folium.LayerControl().add_to(m)
        m.save("map.html")


class Preprocessing(utils.Preprocessing):
    """
    Some interesting attributes, other attributes can be computed by these attributes or have too many Null in samples.

    Keys of Fact Table
        iso_code	               ISO 3166-1 alpha-3 – three-letter country codes
        continent	               Continent of the geographical location
        location	               Geographical location
        date	                   Date of observation
    Confirmed cases
        total_cases         	   Total confirmed cases of COVID-19
    Confirmed deaths
        total_deaths	           Total deaths attributed to COVID-19
    Hospital & ICU
        icu_patients	           Number of COVID-19 patients in intensive care units (ICUs) on a given day
        hosp_patients	           Number of COVID-19 patients in hospital on a given day
    Policy responses
        stringency_index	       Government Response Stringency Index, rescaled to a value from 0 to 100 (100 = strictest response)
    Reproduction rate
        reproduction_rate	       Real-time estimate of the effective reproduction rate (R) of COVID-19. See https://github.com/crondonm/TrackingR/tree/main/Estimates-Database
    Tests & positivity
        total_tests	               Total tests for COVID-19
        positive_rate	           The share of COVID-19 tests that are positive, given as a rolling 7-day average (this is the inverse of tests_per_case)
    Vaccinations
        total_vaccinations	       Total number of COVID-19 vaccination doses administered
        people_vaccinated	       Total number of people who received at least one vaccine dose
        people_fully_vaccinated    Total number of people who received all doses prescribed by the vaccination protocol
        total_boosters	           Total number of COVID-19 vaccination booster doses administered (doses administered beyond the number prescribed by the vaccination protocol)
    Others
        population	               Population (latest available values). See https://github.com/owid/covid-19-data/blob/master/scripts/input/un/population_latest.csv for full list of sources
        population_density   	   Number of people divided by land area, measured in square kilometers, most recent year available
        median_age	               Median age of the population, UN projection for 2020
        aged_65_older	           Share of the population that is 65 years and older, most recent year available
        aged_70_older	           Share of the population that is 70 years and older in 2015
        gdp_per_capita	           Gross domestic product at purchasing power parity (constant 2011 international dollars), most recent year available
        extreme_poverty	           Share of the population living in extreme poverty, most recent year available since 2010
        cardiovasc_death_rate	   Death rate from cardiovascular disease in 2017 (annual number of deaths per 100,000 people)
        diabetes_prevalence  	   Diabetes prevalence (% of population aged 20 to 79) in 2017
        female_smokers	           Share of women who smoke, most recent year available
        male_smokers        	   Share of men who smoke, most recent year available
        handwashing_facilities	   Share of the population with basic handwashing facilities on premises, most recent year available
        hospital_beds_per_thousand Hospital beds per 1,000 people, most recent year available since 2010
        life_expectancy	           Life expectancy at birth in 2019
        human_development_index	   A composite index measuring average achievement in three basic dimensions of human development—a long and healthy life, knowledge and a decent standard of living. Values for 2019, imported from http://hdr.undp.org/en/indicators/137506
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

        self.dataset = df
        self.dataset.info()  # TODO: remove this line

        self.group_attr = "iso_code"
        self.time_attr = "date"

        self.time_series_attributes = [
            "total_cases", "total_deaths",
            "icu_patients", "hosp_patients",
            "reproduction_rate",
            "stringency_index",
            "total_tests", "positive_rate",
            "total_vaccinations", "people_vaccinated", "people_fully_vaccinated", "total_boosters"
        ]
        self.time_free_attributes = [
            "population", "population_density",
            "median_age", "aged_65_older", "aged_70_older",
            "gdp_per_capita", "extreme_poverty",
            "cardiovasc_death_rate", "diabetes_prevalence",
            "female_smokers", "male_smokers",
            "handwashing_facilities", "hospital_beds_per_thousand",
            "life_expectancy", "human_development_index"
        ]

        iso_code = list(pd.read_csv("src/countries_codes_and_coordinates.csv")["Alpha-3 code"].values)
        self.iso_code = [code.replace(' ', '').replace('"', '') for code in iso_code]
        self.dataset = self.data_reduction(self.dataset)

        time_series_data = self.data_transformation(self.dataset)
        time_free_data = self.noise_reduction(self.dataset)
        self.dataset = self.data_integration(time_series_data, time_free_data)

        self.drop_missing_data()
        self.dataset = self.fill_missing_data(self.dataset)
        self.dataset.info()  # TODO: remove this line

    def __call__(self, samples, *args, **kwargs):
        samples = self.data_reduction(samples)

        time_series_data = self.data_transformation(samples)
        time_free_data = self.noise_reduction(samples)
        samples = self.data_integration(time_series_data, time_free_data)

        return self.fill_missing_data(samples)

    def data_reduction(self, samples):
        # filter non-locations, such as continents and international organizes, etc...
        # Remove relevant samples, they can be counted by other samples.
        samples = samples.query("iso_code == @self.iso_code")

        # Remove same repeat attributes, they can be computed by other attributes.
        return samples[[self.group_attr, self.time_attr] + self.time_series_attributes + self.time_free_attributes]

    def drop_missing_data(self):
        """ List all frequency(>95%) attributes, then remove the samples with Null data on these attributes.
            * This method should be used in dataset preprocessing
            * Don't use it to preprocess input samples.

            Remove those locations which have missing data in attribute `population` or `population_density`.
        """
        super().drop_missing_data()

        special_data = self.dataset[["population", "population_density"]]
        invalid_rows = special_data[special_data.isnull().any(axis=1)]
        self.dataset = self.dataset.drop(invalid_rows.index)

    @property
    def fill_data(self):
        if self._fill_data.__len__() == 0:
            for attr in self.time_free_attributes:
                series = self.dataset[attr]
                if attr in ["population", "population_density"]:
                    assert not self.dataset[attr].isnull().values.any()
                # For poor locations
                elif attr == "extreme_poverty":
                    self._fill_data[attr] = (series / self.dataset["population"]).max()
                # For poor locations
                else:
                    self._fill_data[attr] = (series / self.dataset["population"]).min()
        return self._fill_data

    @classmethod
    def unit_test(cls):
        cls(pd.read_csv("src/owid-covid-data.csv"))


class Mining:
    def __init__(self, df: pd.DataFrame):
        self.dataset = df

    def clustering(self, method):
        """
        :param method: K-means, DBSCAN, etc
        :return:
        """
        pass

    def classification(self):
        """
        :return:
            * confusion matrix
            * ROC curve
        """
        # Binning for generating item set
        attributes = [
            "population_density",
            "median_age", "aged_65_older", "aged_70_older",
            "gdp_per_capita", "extreme_poverty",
            "cardiovasc_death_rate", "diabetes_prevalence",
            "female_smokers", "male_smokers",
            "handwashing_facilities", "hospital_beds_per_thousand",
            "life_expectancy", "human_development_index"
        ]
        features = self.dataset[attributes].values
        features = features > np.median(features, axis=0, keepdims=True)

        labels = np.array([np.mean(values) // 10 for values in self.dataset["stringency_index"].values])

        random_indices = np.arange(labels.__len__())

        x_train = features[random_indices[labels.__len__() // 6:], :]
        y_train = labels[random_indices[labels.__len__() // 6:]]

        x_test = features[random_indices[:labels.__len__() // 6], :]
        y_test = labels[random_indices[:labels.__len__() // 6]]

        decision_tree = tree.DecisionTreeClassifier(max_depth=16, max_leaf_nodes=300)
        decision_tree.fit(x_train, y_train)
        print("Train accuracy: {} %".format(round(decision_tree.score(x_train, y_train) * 100, 2)))
        # print("Test accuracy: {} %".format(round(decision_tree.score(x_test, y_test) * 100, 2)))

    def regression(self, features=None, labels=None, k=10):
        # shuffle examples
        n_samples = self.dataset.index.__len__()
        random_indices = np.arange(n_samples)
        np.random.shuffle(random_indices)
        features, labels = features[random_indices], labels[random_indices]

        # k-fold cross validation
        r2_scores = []
        for i in range(k):
            x_train = np.roll(features, i * n_samples // k, axis=0)[n_samples // k:]
            y_train = np.roll(labels, i * n_samples // k, axis=0)[n_samples // k:]

            x_validation = np.roll(features, i * n_samples // k, axis=0)[:-n_samples // k]
            y_validation = np.roll(labels, i * n_samples // k, axis=0)[:-n_samples // k]

            r2_score = []
            for _y_train, _y_validation in zip(y_train.T, y_validation.T):
                model = RandomForestRegressor()
                model.fit(x_train, _y_train)
                r2_score.append(model.score(x_validation, _y_validation))
            r2_scores.append(r2_score)
        k_folds = np.argmax(np.array(r2_scores), axis=0)
        print("Best R2 score:\n", np.max(np.array(r2_scores), axis=0))  # TODO: remove this line

        # extract best models
        models = []
        for col, i in enumerate(k_folds):
            x_train = np.roll(features, i * n_samples // k, axis=0)[n_samples // k:]
            y_train = np.roll(labels, i * n_samples // k, axis=0)[n_samples // k:]

            models.append(RandomForestRegressor())
            models[-1].fit(x_train, y_train[:, col])
        return lambda f: np.array([m.predict(f) for m in models])

    def max_value_regression(self, k=10):
        """Predict the `max(time_series_feature)/population` with inputs `time_free_labels`."""
        feature_attributes = [
            "population_density",
            "median_age", "aged_65_older", "aged_70_older",
            "gdp_per_capita", "extreme_poverty",
            "cardiovasc_death_rate", "diabetes_prevalence",
            "female_smokers", "male_smokers",
            "handwashing_facilities", "hospital_beds_per_thousand",
            "life_expectancy", "human_development_index"
        ]
        features = self.dataset[feature_attributes].values

        label_attributes = [
            "total_cases", "total_deaths",
            "icu_patients", "hosp_patients",
            "reproduction_rate",
            # "stringency_index",
            "total_tests", "positive_rate",
            "total_vaccinations", "people_vaccinated", "people_fully_vaccinated", "total_boosters"
        ]
        labels = dict()
        for attr in label_attributes:
            labels[attr] = [np.max(vs) / p
                            for vs, p in zip(self.dataset[attr].values, self.dataset["population"].values)]
        labels = pd.DataFrame(labels, index=self.dataset.index).values

        return self.regression(features=features, labels=labels, k=k)

    def max_gradient_regression(self, k=10, time_stride=30):
        """Predict the `grad(time_series_feature)/population` with inputs `time_free_labels`."""
        feature_attributes = [
            "population_density",
            "median_age", "aged_65_older", "aged_70_older",
            "gdp_per_capita", "extreme_poverty",
            "cardiovasc_death_rate", "diabetes_prevalence",
            "female_smokers", "male_smokers",
            "handwashing_facilities", "hospital_beds_per_thousand",
            "life_expectancy", "human_development_index"
        ]
        features = self.dataset[feature_attributes].values

        label_attributes = [
            "total_cases", "total_deaths",
            "icu_patients", "hosp_patients",
            "reproduction_rate",
            # "stringency_index",
            "total_tests", "positive_rate",
            "total_vaccinations", "people_vaccinated", "people_fully_vaccinated", "total_boosters"
        ]
        labels = dict()
        for attr in label_attributes:
            labels[attr] = [np.max((vs[time_stride:] - vs[:-time_stride]) / time_stride) / p
                            for vs, p in zip(self.dataset[attr].values, self.dataset["population"].values)]
        labels = pd.DataFrame(labels, index=self.dataset.index).values

        return self.regression(features=features, labels=labels, k=k)

    def association_mining(self):
        # Binning for generating item set
        attributes = [
            "population_density",
            "median_age", "aged_65_older", "aged_70_older",
            "gdp_per_capita", "extreme_poverty",
            "cardiovasc_death_rate", "diabetes_prevalence",
            "female_smokers", "male_smokers",
            "handwashing_facilities", "hospital_beds_per_thousand",
            "life_expectancy", "human_development_index"
        ]
        dataset = self.dataset[attributes].values
        dataset = dataset > np.mean(dataset, axis=0, keepdims=True)  # the set of items that is greater than mean value

        # Generate dataset
        examples = []
        for masks in dataset:
            example = [attr for attr, mask in zip(attributes, masks) if mask]
            if example:
                examples.append(example)

        utils.AssociationMining().fit(
            data=examples,
            min_support=0.4,
            min_confidence=0.5
        )

    def sequential_pattern_mining(self):
        # Binning for generating sequent
        attributes = [
            "total_cases", "total_deaths",
            "icu_patients", "hosp_patients",
            "reproduction_rate",
            # "stringency_index",
            "total_tests", "positive_rate",
            "total_vaccinations", "people_vaccinated", "people_fully_vaccinated", "total_boosters"
        ]
        dataset = np.array([[seq for seq in seq_list] for seq_list in self.dataset[attributes].values])
        grad_dataset = dataset[:, :, 7:] - dataset[:, :, :-7]
        masks = grad_dataset[:, :, :-90:90] > grad_dataset[:, :, 90::90]  # the set of items that the new_count is increasing

        # Generate dataset
        examples = []
        for mask_seq in masks:
            example = []
            for i in range(attributes.__len__()):
                item_set = set([attributes[time] for time, mask in enumerate(mask_seq[i, :]) if mask])
                if item_set != set():
                    example.append(item_set)
            if example:
                examples.append(example)

        utils.SequentialPatternMining().fit(
            data=examples,
            min_support=100,
        )

    @classmethod
    def unit_test(cls):
        print("=" * 16 + " data preprocessing " + "=" * 16)
        dataset = Preprocessing(pd.read_csv("src/owid-covid-data.csv")).dataset

        print("=" * 16 + " classification " + "=" * 16)
        cls(dataset).classification()

        # print("=" * 16 + " regression " + "=" * 16)
        # cls(dataset).max_value_regression()
        # cls(dataset).max_gradient_regression()
        #
        # print("=" * 16 + " association mining " + "=" * 16)
        # cls(dataset).association_mining()
        #
        # print("=" * 16 + " sequential pattern mining " + "=" * 16)
        # cls(dataset).sequential_pattern_mining()


if __name__ == "__main__":
    # MatplotlibTimeSeriesVisualization.unit_test()
    # MatplotlibTimeFreeVisualization.unit_test()
    # MapVisualization.unit_test()

    # Preprocessing.unit_test()
    Mining.unit_test()
