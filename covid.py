import numpy as np
import pandas as pd
import folium

import utils

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


class MatplotlibTimeSeriesVisualization(utils.MatplotlibTimeSeriesVisualization):

    @classmethod
    def unit_test(cls):
        locations = [item["properties"]["name"] for item in pd.read_json("src/world-countries.json")["features"]]
        dataset = pd.read_csv("src/owid-covid-data.csv")
        dataset = dataset.query("location == @locations")

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
        location = [item["properties"]["name"] for item in pd.read_json("src/world-countries.json")["features"]]
        dataset = pd.read_csv("src/owid-covid-data.csv")
        dataset = dataset.query("location == @location")

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


class MapVisualization(utils.MapVisualization):

    @classmethod
    def unit_test(cls):
        location = [item["properties"]["name"] for item in pd.read_json("src/world-countries.json")["features"]]

        temp_df = [[loc, np.random.rand() + 1] for loc in location]
        dataset = pd.DataFrame(data=temp_df, columns=["locations", "value"])

        m = folium.Map(location=[0, 0], zoom_start=3)
        folium.Choropleth(
            geo_data="src/world-countries.json",
            name="choropleth",
            data=dataset,
            columns=["locations", "value"],
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

        self.group_attr = "location"
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

        self._locations = [item["properties"]["name"] for item in pd.read_json("src/world-countries.json")["features"]]
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
        samples = samples.query("location == @self._locations")

        # Remove same repeat attributes, they can be computed by other attributes.
        return samples[["location", "date"] + self.time_series_attributes + self.time_free_attributes]

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

        # location = [item["properties"]["name"] for item in pd.read_json("src/world-countries.json")["features"]]
        dataset = pd.read_csv("src/owid-covid-data.csv")
        # location_categories = set(list(dataset["location"].value_counts().index.values))

        # dataset = dataset.query("location == @location")
        # valid_location_categories = set(list(dataset["location"].value_counts().index.values))
        #
        # print("ans = \n", location_categories - valid_location_categories)
        cls(pd.read_csv("src/owid-covid-data.csv"))


if __name__ == "__main__":
    # MatplotlibTimeSeriesVisualization.unit_test()
    # MatplotlibTimeFreeVisualization.unit_test()
    # MapVisualization.unit_test()

    Preprocessing.unit_test()
