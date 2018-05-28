import numpy as np
import train_model

N_COMPONENTS = train_model.N_COMPONENTS

class Patient:
    # a static value to record the number of event_type in state_proportions
    vol = 0;
    trend_types = [];
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.state_proportions = dict()
        self.trends = dict()
        self.all_proportions = []
        self.events = dict()
        self.events_type = ["mbs", "pbs"]

    def __str__(self):
        return "patient_id: %s \nvolumn: %s \nproportions: %s \ntrends: %s" % (
            self.patient_id, self.vol, self.all_proportions, self.trends
        )

    def add_events(self, event, type):
        self.events[type] = event

    def add_state_proportion(self, state_proportion, col_index):
        self.state_proportions[col_index] = state_proportion

    def add_trend(self, trend, type):
        self.trends[type] = trend

    def has_nan_var(self):
        for i in range(self.vol):
            if i not in self.state_proportions:
                self.state_proportions[i] = np.zeros((N_COMPONENTS)**2)
                # return True
        for i in self.events_type:
            if i not in self.events:
                self.events[i] = np.zeros(1)
        for t in self.trend_types:
            if t not in self.trends:
                return True
        # for i in self.events_type:
        #     for j in self.events[i]:
        #         if j>=90:
        #             return True
        return False

    def concate_proportions(self):
        for i in range(self.vol):
            self.all_proportions.append(self.state_proportions[i])
        self.all_proportions = np.asarray(self.all_proportions).flatten()

    def get_proportions(self):
        return self.all_proportions

    def get_trends_by_key(self, key):
        return self.trends[key]
