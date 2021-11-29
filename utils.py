import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import folium


class TimeSeriesVisualization:
    def __init__(self, cache_file, **kwargs):
        try:
            with open(cache_file, "rb") as cache:
                self.segments, self.attributes, self.date_categories = pickle.load(cache)
        except FileNotFoundError:
            self.segments, self.attributes, self.date_categories = self.time_query(**kwargs)
            with open(cache_file, "wb") as cache:
                pickle.dump((self.segments, self.attributes, self.date_categories), cache)

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
                boundary = [np.min(x), np.max(x), np.min(y), np.max(y)]
                segments.append({"category": cat, "attribute": attr, "x": x, "y": y, "boundary": boundary})

        return segments, attributes, date_categories


class MatplotlibTimeSeriesVisualization(TimeSeriesVisualization):
    def __init__(self, cache_file, **kwargs):
        super().__init__(cache_file, **kwargs)

        # ---------------- Initial the line segments ----------------
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.3, bottom=0.25)
        self.ax.grid(True)
        ticks = [i for i, d in enumerate(self.date_categories) if d[8:10] == "01"]
        labels = [self.date_categories[i][:7] for i in ticks]
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels(labels)
        for label in self.ax.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')
        for seg in self.segments:
            self.ax.plot(seg["x"], seg["y"], alpha=0.1)

        # ---------------- Slider ----------------
        sax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        sax_max = 31
        self.slider = Slider(sax, 'Stride', valmin=1, valmax=sax_max, valinit=0)
        self.slider.on_changed(self.call_slider)

        # ---------------- RadioButtons ----------------
        rax_w = 0.04 + 0.008 * np.max([attr.__len__() for attr in self.attributes])
        rax_h = 0.04 * self.attributes.__len__()
        rax = plt.axes([0.025, 0.5, rax_w, rax_h], facecolor='lightgoldenrodyellow')
        self.radio_buttons = RadioButtons(rax, self.attributes, active=0)
        for label in self.radio_buttons.labels:
            label.set_size(8)
        self.radio_buttons.on_clicked(self.call_radio_buttons)
        self.call_radio_buttons(None)

        # ---------------- Motion Notify Event ----------------
        self.fig.canvas.mpl_connect('motion_notify_event', self.call_motion_notify)

        plt.show()

    def call_radio_buttons(self, event):
        self._lock = True
        self.slider.set_val(0)
        self.slider.valtext.set_text("0")
        self._lock = False

        x_min = x_max = y_min = y_max = 0
        for i, line in enumerate(self.ax.lines):
            if self.segments[i]["attribute"] == self.radio_buttons.value_selected:
                line.set_alpha(1.0)
                x_min = min(x_min, self.segments[i]["boundary"][0])
                x_max = max(x_max, self.segments[i]["boundary"][1])
                y_min = min(y_min, self.segments[i]["boundary"][2])
                y_max = max(y_max, self.segments[i]["boundary"][3])
            else:
                line.set_alpha(0.05)
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.fig.canvas.draw_idle()

    _lock = False

    def call_slider(self, event):
        if not self._lock:
            # set the slider value be integer
            self._lock = True
            stride = int(round(self.slider.val))
            self.slider.set_val(stride)
            self.slider.valtext.set_text(str(stride))
            self._lock = False

            if stride > 0:
                x_min = x_max = y_min = y_max = 0
                for line, seg in zip(self.ax.lines, self.segments):
                    if seg["y"].__len__() <= stride:
                        continue
                    if seg["attribute"] != self.radio_buttons.value_selected:
                        continue
                    ys = (seg["y"][stride:] - seg["y"][:-stride]) / stride
                    xs = seg["x"][stride:]
                    x_min = min(x_min, np.min(xs))
                    x_max = max(x_max, np.max(xs))
                    y_min = min(y_min, np.min(ys))
                    y_max = max(y_max, np.max(ys))

                    line.set_xdata(xs)
                    line.set_ydata(ys)

                self.ax.set_xlim(x_min, x_max)
                self.ax.set_ylim(y_min, y_max)
                self.fig.canvas.draw_idle()

    def call_motion_notify(self, event):
        if event.inaxes:
            self.ax.collections, self.ax.texts = [], []

            attribute = self.radio_buttons.value_selected
            segments = [seg for seg in self.segments if seg["attribute"] == attribute]
            curves = [line for line, seg in zip(self.ax.lines, self.segments) if seg["attribute"] == attribute]
            indices = [np.argmin(np.abs(event.xdata - line.get_xdata())) for line in curves]
            xs = [line.get_xdata()[index] for index, line in zip(indices, curves)]
            ys = [line.get_ydata()[index] for index, line in zip(indices, curves)]

            indices = np.argsort(ys)[-5:]  # TODO: convert the threshold `5` to a parameter.
            xs = [xs[index] for index in indices]
            ys = [ys[index] for index in indices]
            texts = [segments[index]["category"] for index in indices]
            colors = [curves[index].get_color() for index in indices]
            for x, y, t, c in zip(xs, ys, texts, colors):
                self.ax.text(x + 10, y, t, color=c, verticalalignment="center")  # TODO: convert `10` to a parameter.
            self.ax.scatter(xs, ys, c=colors)

            for line, seg in zip(self.ax.lines, self.segments):
                if seg["attribute"] == self.radio_buttons.value_selected and seg["category"] in texts:
                    line.set_alpha(1.0)
                else:
                    line.set_alpha(0.05)

            self.fig.canvas.draw_idle()


class QtTimeSeriesVisualization(TimeSeriesVisualization):
    pass


class JSTimeSeriesVisualization(TimeSeriesVisualization):
    pass


class TimeFreeVisualization:
    def __init__(self, cache_file, **kwargs):
        try:
            with open(cache_file, "rb") as cache:
                self.stats = pickle.load(cache)
        except FileNotFoundError:
            self.stats = self.stats_query(**kwargs)
            with open(cache_file, "wb") as cache:
                pickle.dump(self.stats, cache)

    @staticmethod
    def stats_query(dataset, group_attr, attributes):
        fact_table = dataset[[group_attr] + attributes]
        return fact_table.groupby([group_attr]).mean()


class MatplotlibTimeFreeVisualization(TimeFreeVisualization):
    def __init__(self, cache_file, **kwargs):
        super().__init__(cache_file, **kwargs)

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.3, bottom=0.25)

        # ---------------- Slider ----------------
        self.slider_ax = plt.axes([0.15, 0.1, 0.55, 0.03])
        self.slider = Slider(ax=self.slider_ax, label="Attributes", valmin=0, valmax=self.stats.columns.__len__() - 1)
        self.slider.on_changed(self.call_slider)

        # ---------------- RadioButtons ----------------
        radio_buttons_labels = ["pie chart", "box plot", "histogram"]
        rax_w = 0.04 + 0.012 * np.max([label.__len__() for label in radio_buttons_labels])
        rax_h = 0.04 * radio_buttons_labels.__len__()
        self.radio_buttons_ax = plt.axes([0.025, 0.5, rax_w, rax_h], facecolor='lightgoldenrodyellow')
        self.radio_buttons = RadioButtons(self.radio_buttons_ax, radio_buttons_labels, active=0)
        self.radio_buttons.on_clicked(self.call_radio_buttons)

        # ---------------- figure resize event ----------------
        self.fig.canvas.mpl_connect('resize_event', self.call_resize)

        # ---------------- initial widgets status ----------------
        self.call_slider(None)
        self.slider_ax.set_visible(False)
        self.call_radio_buttons(None)

        plt.show()

    _lock = False

    def call_radio_buttons(self, event):
        if self.radio_buttons.value_selected == "pie chart":
            self.ax.clear()
            self.slider_ax.set_visible(True)
            self.ax.set_aspect("equal")
            self.ax.set_title("pie chart")

            if not self._lock:
                # set the slider value be integer
                self._lock = True
                integer_val = int(round(self.slider.val))
                self.slider.set_val(integer_val)
                text = self.stats.columns[integer_val]
                self.slider.valtext.set_text(text)
                self._lock = False

                attr = self.slider.valtext.get_text()
                series = self.stats[attr].dropna()
                series = series.nlargest(10, keep="first")
                self.ax.pie(
                    x=series.values,
                    explode=0.1 * (series.values == np.max(series.values)),
                    labels=series.index.tolist(),
                    autopct='%1.1f%%', shadow=True, startangle=90)

        if self.radio_buttons.value_selected == "box plot":
            self.ax.clear()
            self.slider_ax.set_visible(False)
            self.ax.yaxis.grid(True)
            self.ax.set_aspect("auto")
            self.ax.set_title("box plot")

            box_data = [self.stats[attr].dropna() for attr in self.stats.columns]
            box_data = [(df - df.min()) / (df.max() - df.min()) for df in box_data]
            self.ax.boxplot(box_data, notch=True, vert=True, patch_artist=True, labels=self.stats.columns)
            for label in self.ax.get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')

        if self.radio_buttons.value_selected == "histogram":
            self.ax.clear()
            self.slider_ax.set_visible(False)
            self.ax.yaxis.grid(True)
            self.ax.set_aspect("auto")
            self.ax.set_title("histogram")

            hist_data = [self.stats[attr].dropna() for attr in self.stats.columns]
            hist_data = [(df - df.min()) / (df.max() - df.min()) for df in hist_data]
            _, _, bars = self.ax.hist(hist_data)
            colors = [patches[0].get_facecolor() for patches in bars]
            handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
            font_size = int(self.fig.get_size_inches()[1])
            self.ax.legend(handles, self.stats.columns, loc='upper right', prop={"size": font_size})

        self.fig.canvas.draw_idle()

    def call_slider(self, event):
        if self.radio_buttons.value_selected == "pie chart":
            self.call_radio_buttons(None)

    def call_resize(self, event):
        if self.radio_buttons.value_selected == "histogram":
            self.call_radio_buttons(None)


class MapVisualization:
    def __init__(self, cache_file, **kwargs):
        pass


class Preprocessing:
    _group_categories = None

    @property
    def group_categories(self):
        if self._group_categories is None:
            self._group_categories = sorted(list(self.dataset[self.group_attr].value_counts().index.values))
        return self._group_categories

    _time_categories = None

    @property
    def time_categories(self):
        if self._time_categories is None:
            self._time_categories = sorted(list(self.dataset[self.time_attr].value_counts().index.values))
        return self._time_categories

    def __init__(self, df: pd.DataFrame):
        self.dataset = df

        self.group_attr = None
        self.time_attr = None

        self.time_series_attributes = None
        self.time_free_attributes = None

    def __call__(self, samples, *args, **kwargs):
        pass

    def data_reduction(self, samples):
        # Remove same repeat attributes, they can be computed by other attributes.
        return samples[["location", "date"] + self.time_series_attributes + self.time_free_attributes]

    def data_transformation(self, samples):
        """Synchronize the time-based attributes"""
        time_series_data = {attr: [] for attr in [self.group_attr] + self.time_series_attributes}
        for group_cat in self.group_categories:
            # query the location data by attribute `location`
            group_data = samples[samples[self.group_attr] == group_cat]

            # generate the full data of a given location.
            time_series_data[self.group_attr].append(group_cat)
            for attr in self.time_series_attributes:
                series = group_data[[self.time_attr, attr]].dropna()
                xp = np.array([self.time_categories.index(date) for date in series[self.time_attr].values])
                fp = series[attr].values.flatten()
                try:
                    # For the missing data we fill with the interpolation values
                    time_series_data[attr].append(np.interp(np.arange(self.time_categories.__len__()), xp, fp))
                except ValueError:
                    assert xp.__len__() == fp.__len__()
                    time_series_data[attr].append(None)
        time_series_data = pd.DataFrame(time_series_data)
        return time_series_data.set_index([self.group_attr])

    def noise_reduction(self, samples):
        """Process the time-free attributes"""
        group_data = samples[[self.group_attr] + self.time_free_attributes]
        return group_data.groupby([self.group_attr]).mean()

    @staticmethod
    def data_integration(time_series_data, time_free_data):
        """Data Warehouse: Star Schema"""
        return pd.merge(time_series_data, time_free_data, left_index=True, right_index=True)

    def drop_missing_data(self):
        """ List all frequency(>95%) attributes, then remove the samples with Null data on these attributes.
            * This method should be used in dataset preprocessing
            * Don't use it to preprocess input samples.
        """
        count = self.dataset.count()
        # List all frequency(>95%) attributes, then remove samples with Null data on these attributes.
        frequency_attributes = count[count > 0.95 * np.max(count.values)].index
        frequency_dataset = self.dataset[frequency_attributes]
        invalid_rows = frequency_dataset[frequency_dataset.isnull().any(axis=1)]
        self.dataset = self.dataset.drop(invalid_rows.index)

    _fill_data = dict()

    @property
    def fill_data(self):
        """Compute values for fill missing data by using the original dataset."""
        return self._fill_data

    def fill_missing_data(self, samples):
        """This method relay on the class property `self.fill_data`."""
        # ----------- Fill missing time-series data -----------
        fill = np.zeros(shape=(self.time_categories.__len__(),))
        for attr in self.time_series_attributes:
            values = [cell if isinstance(cell, np.ndarray) else fill for cell in samples.values]
            samples.update(pd.DataFrame({attr: values}, index=samples.index))

        # ----------- Fill missing time-free data -----------
        for attr in self.fill_data.keys():
            samples[attr] = samples[attr].fillna(value=self.fill_data[attr])

        return samples

    def continuous2discrete(self):
        pass

    def discrete2continuous(self):
        pass


class Mining:
    def __init__(self, df: pd.DataFrame):
        pass

    def clustering(self, method):
        """
        :param method: K-means, DBSCAN, etc
        :return:
        """
        pass

    def classification(self, method):
        """
        :param method: Naive Bayesian Classification, Decision Tree, NeuralNetwork, etc
        :return:
            * confusion matrix
            * ROC curve
        """
        pass

    def regression(self, method):
        """
        :return: method: piecewise OLS, Random Forests, etc
        """
        pass

    def association_mining(self):
        pass


class Postprocessing:
    def __init__(self, df: pd.DataFrame):
        pass

    def map_visualization(self):
        import pandas as pd
        import folium

        m = folium.Map(location=[0, 0], zoom_start=3)
        folium.Choropleth(
            geo_data="src/us-states.json",
            name="choropleth",
            data=pd.read_csv("src/US_Unemployment_Oct2012.csv"),
            columns=["State", "Unemployment"],
            key_on="feature.id",
            fill_color="OrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="total_cases",
        ).add_to(m)
        folium.LayerControl().add_to(m)
        m.save("index.html")
        pass

    def applications(self):
        pass


if __name__ == "__main__":
    print(0)
