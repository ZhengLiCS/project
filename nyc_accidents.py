import numpy as np
import pandas as pd

import utils


dataset = pd.read_csv("src/NYC Accidents 2020.csv")
print(dataset)
dataset.info()
print(dataset.describe())
print(dataset[["CONTRIBUTING FACTOR VEHICLE 1", "CONTRIBUTING FACTOR VEHICLE 2"]])
print(dataset[["VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2"]])


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

        # self.group_attr = "iso_code"
        # self.time_attr = "date"

        self.time_series_attributes = [
            "CRASH DATE", "CRASH TIME",
            "BOROUGH",
            "LATITUDE", "LONGITUDE", "LOCATION",
            "ON STREET NAME", "CROSS STREET NAME", "OFF STREET NAME",
            "NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED", "NUMBER OF PEDESTRIANS INJURED", "NUMBER OF PEDESTRIANS KILLED", "NUMBER OF CYCLIST INJURED", "NUMBER OF CYCLIST KILLED", "NUMBER OF MOTORIST INJURED", "NUMBER OF MOTORIST KILLED",
            "CONTRIBUTING FACTOR VEHICLE 1", "CONTRIBUTING FACTOR VEHICLE 2", "CONTRIBUTING FACTOR VEHICLE 3", "CONTRIBUTING FACTOR VEHICLE 4", "CONTRIBUTING FACTOR VEHICLE 5",
            "VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3", "VEHICLE TYPE CODE 4", "VEHICLE TYPE CODE 5",
        ]
        self.time_free_attributes = []

        self.dataset = self.dataset[self.time_series_attributes]

        set_dataset = dict()

        attributes = [
            "CONTRIBUTING FACTOR VEHICLE 1", "CONTRIBUTING FACTOR VEHICLE 2", "CONTRIBUTING FACTOR VEHICLE 3",
            "CONTRIBUTING FACTOR VEHICLE 4", "CONTRIBUTING FACTOR VEHICLE 5",
        ]
        series = self.dataset[attributes].fillna("NULL")
        values = [set(items) - {"NULL"} for items in series.values]
        set_dataset["CONTRIBUTING FACTOR VEHICLE"] = values
        invalid_indices = [i for i, items in enumerate(values) if items == set()]

        attributes = [
            "VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3", "VEHICLE TYPE CODE 4",
            "VEHICLE TYPE CODE 5"
        ]
        series = self.dataset[attributes].fillna("NULL")
        values = [set(items) - {"NULL"} for items in series.values]
        set_dataset["VEHICLE TYPE CODE"] = values
        invalid_indices += [i for i, items in enumerate(values) if items == set()]
        # print(invalid_indices)

        pd.DataFrame(set_dataset).info()
        set_dataset = pd.DataFrame(set_dataset).drop(index=invalid_indices)
        set_dataset.info()

        # invalid_indices =
        #
        # set_dataset = pd.DataFrame(set_dataset)
        #
        # print("------------")
        # print(set_dataset)
        # print(temp_dataset)
        # print(temp_dataset.values)
        # for item in temp_dataset.data():
        #     print(item)

        # iso_code = list(pd.read_csv("src/countries_codes_and_coordinates.csv")["Alpha-3 code"].values)
        # self.iso_code = [code.replace(' ', '').replace('"', '') for code in iso_code]
        # self.dataset = self.data_reduction(self.dataset)
        #
        # time_series_data = self.data_transformation(self.dataset)
        # time_free_data = self.noise_reduction(self.dataset)
        # self.dataset = self.data_integration(time_series_data, time_free_data)
        #
        # self.drop_missing_data()
        # self.dataset = self.fill_missing_data(self.dataset)
        # self.dataset.info()  # TODO: remove this line

    def __call__(self, samples, *args, **kwargs):
        samples = self.data_reduction(samples)

        time_series_data = self.data_transformation(samples)
        time_free_data = self.noise_reduction(samples)
        samples = self.data_integration(time_series_data, time_free_data)

        return self.fill_missing_data(samples)

    def data_reduction(self, samples):
        # filter useless attributes, such as `zip code` and `COLLISION_ID`, etc...
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
        set_attributes = [
            "CONTRIBUTING FACTOR VEHICLE 1", "CONTRIBUTING FACTOR VEHICLE 2", "CONTRIBUTING FACTOR VEHICLE 3", "CONTRIBUTING FACTOR VEHICLE 4", "CONTRIBUTING FACTOR VEHICLE 5",
            "VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3", "VEHICLE TYPE CODE 4", "VEHICLE TYPE CODE 5"
        ]

        temp_dataset = self.dataset[set_attributes].fillna("NULL")
        # temp_dataset =

        # if self._fill_data.__len__() == 0:
        #     for attr in self.time_free_attributes:
        #         series = self.dataset[attr]
        #         if attr in ["population", "population_density"]:
        #             assert not self.dataset[attr].isnull().values.any()
        #         # For poor locations
        #         elif attr == "extreme_poverty":
        #             self._fill_data[attr] = (series / self.dataset["population"]).max()
        #         # For poor locations
        #         else:
        #             self._fill_data[attr] = (series / self.dataset["population"]).min()
        # return self._fill_data

    @classmethod
    def unit_test(cls):
        cls(pd.read_csv("src/NYC Accidents 2020.csv"))


if __name__ == "__main__":
    Preprocessing.unit_test()
