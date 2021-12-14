import pickle
from itertools import combinations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, RadioButtons


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
            values = [cell if isinstance(cell, np.ndarray) else fill for cell in samples[attr].values]
            samples.update(pd.DataFrame({attr: values}, index=samples.index))

        # ----------- Fill missing time-free data -----------
        for attr in self.fill_data.keys():
            samples[attr] = samples[attr].fillna(value=self.fill_data[attr])

        return samples

    def continuous2discrete(self):
        pass

    def discrete2continuous(self):
        pass


class AssociationMining:
    def __init__(self):
        self.data = []
        self.frequency_item_set = []
        self.supports = []

    @staticmethod
    def gen_candidate_item_set(frequency_item_set):
        candidate_item_set = []
        for i in range(frequency_item_set.__len__()):
            for j in range(i + 1, frequency_item_set.__len__()):
                size = frequency_item_set[i].__len__()
                # self-joining
                item_set = frequency_item_set[i] | frequency_item_set[j]
                if item_set.__len__() != size + 1:
                    continue
                # pruning
                try:
                    for item in combinations(item_set, size):
                        assert set(item) in frequency_item_set
                    candidate_item_set.append(item_set)
                except AssertionError:
                    pass
        return candidate_item_set

    def gen_frequency_item_set(self, candidate_item_set, min_support):
        supports = np.zeros(shape=(candidate_item_set.__len__(), ))
        for i, item in enumerate(candidate_item_set):
            for data_item in self.data:
                supports[i] += item.issubset(data_item)
        supports /= self.data.__len__()
        # scan
        valid_indices = [i for i, s in enumerate(supports) if s >= min_support]
        return [candidate_item_set[i] for i in valid_indices], [supports[i] for i in valid_indices]

    def fit(self, data, min_support, min_confidence):
        self.data = [set(item) for item in data]

        # Apriori Algorithm
        candidate_item_set = [set([item]) for item in set(sum(data, []))]
        while candidate_item_set.__len__() != 0:
            frequency_item_set, supports = self.gen_frequency_item_set(candidate_item_set, min_support)
            self.frequency_item_set += frequency_item_set
            self.supports += supports
            candidate_item_set = self.gen_candidate_item_set(frequency_item_set)

        # Association Mining
        rules = []
        for features, support in zip(self.frequency_item_set, self.supports):
            for targets in self.frequency_item_set:
                if features & targets != set():
                    continue
                try:
                    index = self.frequency_item_set.index(features | targets)
                    if self.supports[index] / support >= min_confidence:
                        rules.append([features, targets, self.supports[index], self.supports[index] / support])
                except ValueError:
                    pass

        # Display Message
        try:
            prefix_length = max([str(rule[0]).__len__() for rule in rules])
            suffix_length = max([str(rule[1]).__len__() for rule in rules])
        except ValueError:
            print("Invalid support or confidence!")
            raise ValueError
        message_format = "{:<" + str(prefix_length) + "}    {:<" + str(suffix_length) + "} {} {}"
        print(message_format.format("rule", "", "support", "confidence"))
        message_format = "{:<" + str(prefix_length) + "} -> {:<" + str(suffix_length) + "} {:.2e} {:.2e}"
        for rule in rules:
            print(message_format.format(str(rule[0]), str(rule[1]), rule[2], rule[3]))
        return rules

    @classmethod
    def unit_test(cls):
        """
        Example from CSCI415_15.ppt, Page 11/16:
            {Milk,Diaper} -> {Beer}
                 s=0.4, c=0.67
            {Milk,Beer} -> {Diaper}
                 s=0.4, c=1.0
            {Diaper,Beer} -> {Milk}
                 s=0.4, c=0.67
            {Beer} -> {Milk,Diaper}
                  s=0.4, c=0.67
            {Diaper} -> {Milk,Beer}
                  s=0.4, c=0.5
            {Milk} -> {Diaper,Beer}
                   s=0.4, c=0.5
        """
        AssociationMining().fit(
            data=[
                ["Bread", "Milk"],
                ["Bread", "Diaper", "Beer", "Eggs"],
                ["Milk", "Diaper", "Beer", "Coke"],
                ["Bread", "Milk", "Diaper", "Beer"],
                ["Bread", "Milk", "Diaper", "Coke"]
            ],
            min_support=0.4,
            min_confidence=0.5
        )


class SequentialPatternMining(AssociationMining):
    @staticmethod
    def gen_candidate_item_set(frequency_item_set):
        candidate_item_set = []
        for i in range(frequency_item_set.__len__()):
            for j in range(frequency_item_set.__len__()):
                size = sum([item.__len__() for item in frequency_item_set[i]])
                # <{a}> + <{b}> -> <{a}, {b}>
                if size == 1:
                    # self-joining
                    item_set = frequency_item_set[i] + frequency_item_set[j]
                    # pruning
                    try:
                        assert frequency_item_set[i] in frequency_item_set
                        assert frequency_item_set[j] in frequency_item_set
                        candidate_item_set.append(item_set)
                    except AssertionError:
                        pass

                # <{a}> + <{b}> -> <{a, b}>
                if size == 1:
                    # self-joining
                    if i >= j:
                        continue
                    item_set = [frequency_item_set[i][0] | frequency_item_set[j][0]]
                    # pruning
                    try:
                        assert frequency_item_set[i] in frequency_item_set
                        assert frequency_item_set[j] in frequency_item_set
                        candidate_item_set.append(item_set)
                    except AssertionError:
                        pass

                if size >= 2:
                    item_set = None

                    if frequency_item_set[i][0].__len__() == 1 and frequency_item_set[j][-1].__len__() == 1:
                        if frequency_item_set[i][1:] == frequency_item_set[j][:-1]:
                            item_set = frequency_item_set[i] + [frequency_item_set[j][-1]]

                    if frequency_item_set[i][0].__len__() != 1 and frequency_item_set[j][-1].__len__() == 1:
                        for prefix in frequency_item_set[i][0]:
                            left = [frequency_item_set[i][0] - set(prefix)] + frequency_item_set[i][1:]
                            if left == frequency_item_set[j][:-1]:
                                item_set = frequency_item_set[i] + [frequency_item_set[j][-1]]
                                break

                    if frequency_item_set[i][0].__len__() == 1 and frequency_item_set[j][-1].__len__() != 1:
                        for suffix in frequency_item_set[j][-1]:
                            right = frequency_item_set[j][:-1] + [frequency_item_set[j][-1] - set(suffix)]
                            if frequency_item_set[i][1:] == right:
                                item_set = [frequency_item_set[i][0]] + frequency_item_set[j]
                                break

                    if frequency_item_set[i][0].__len__() != 1 and frequency_item_set[j][-1].__len__() != 1:
                        for prefix in frequency_item_set[i][0]:
                            left = [frequency_item_set[i][0] - set(prefix)] + frequency_item_set[i][1:]
                            for suffix in frequency_item_set[j][-1]:
                                right = frequency_item_set[j][:-1] + [frequency_item_set[j][-1] - set(suffix)]
                                if left == right:
                                    item_set = frequency_item_set[i][:-1] + [frequency_item_set[j][-1]]
                                    break
                            if item_set is not None:
                                break

                    # pruning
                    try:
                        assert item_set is not None
                        assert sum([item.__len__() for item in item_set]) == size + 1
                        assert item_set not in candidate_item_set
                        for set_cursor in range(item_set.__len__()):
                            if item_set[set_cursor].__len__() == 1:
                                down_sample = item_set.copy()
                                down_sample.pop(set_cursor)
                                assert down_sample in frequency_item_set
                            else:
                                for event in item_set[set_cursor]:
                                    down_sample = item_set.copy()
                                    down_sample[set_cursor] = down_sample[set_cursor] - {event}
                                    assert down_sample in frequency_item_set
                        candidate_item_set.append(item_set)
                    except AssertionError:
                        pass

        return candidate_item_set

    def gen_frequency_item_set(self, candidate_item_set, min_support):
        supports = np.zeros(shape=(candidate_item_set.__len__(), ))
        for i, item in enumerate(candidate_item_set):
            for data_item in self.data:
                item_cursor, data_item_cursor = 0, 0
                while data_item_cursor != data_item.__len__():
                    if item[item_cursor].issubset(data_item[data_item_cursor]):
                        item_cursor += 1
                    data_item_cursor += 1
                    if item_cursor == item.__len__() and data_item_cursor <= data_item.__len__():
                        supports[i] += 1
                        break
        valid_indices = [i for i, s in enumerate(supports) if s >= min_support]
        return [candidate_item_set[i] for i in valid_indices], [supports[i] for i in valid_indices]

    def fit(self, data, min_support, *args):
        self.data = data

        # Apriori Algorithm
        events = set(sum([sum([list(evs) for evs in item], []) for item in data], []))
        candidate_item_set = [[{event}]for event in events]
        while candidate_item_set.__len__() != 0:
            frequency_item_set, supports = self.gen_frequency_item_set(candidate_item_set, min_support)
            self.frequency_item_set += frequency_item_set
            self.supports += supports
            candidate_item_set = self.gen_candidate_item_set(frequency_item_set)

        # Display Message
        length = max([str(pattern).__len__() for pattern in self.frequency_item_set])
        message_format = "{:<" + str(length) + "} {}"
        print(message_format.format("sequential pattern", "support"))
        message_format = "{:<" + str(length) + "} {:.2e}"
        for sequential_pattern, support in zip(self.frequency_item_set, self.supports):
            print(message_format.format(str(sequential_pattern), support))

        return self.frequency_item_set

    @classmethod
    def unit_test(cls):
        """
        Example from CSCI415_21.ppt, Page 41/43:
            +---------+-----------------+
            | Seq. ID | Sequence        |
            +---------+-----------------+
            | 10      | <(be)(ce)d>     |
            | 20      | <(ah)(bf)abf>   |
            | 30      | <(bf)(ce)b(fg)> |
            | 40      | <(bd)cb(ac)>    |
            | 50      | <a(bd)bcb(ade)> |
            +---------+-----------------+
        """
        SequentialPatternMining().fit(
            data=[
                [{"b", "d"}, {"c"}, {"b"}, {"a", "c"}],
                [{"b", "f"}, {"c", "e"}, {"b"}, {"f", "g"}],
                [{"a", "h"}, {"b", "f"}, {"a"}, {"b"}, {"f"}],
                [{"b", "e"}, {"c", "e"}, {"d"}],
                [{"a"}, {"b", "d"}, {"b"}, {"c"}, {"b"}, {"a", "d", "e"}]
            ],
            min_support=2
        )


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
    AssociationMining.unit_test()
    SequentialPatternMining.unit_test()
