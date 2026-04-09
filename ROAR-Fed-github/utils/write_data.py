import os
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column

class Metrics(object):
	def __init__(self, save_path, plot_path):
		self.save_path = save_path
		self.plot_path = plot_path
		self.figures = []
		self.results = None

	def add(self, **kwargs):
		df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
		if self.results is None:
			self.results = df
		else:
			self.results = self.results.append(df, ignore_index=True)

	def save(self, title='Training Results'):
		if len(self.figures) > 0:
			if os.path.isfile(self.plot_path):
				os.remove(self.plot_path)
			output_file(self.plot_path, title=title)
			plots = column(*self.figures)
			save(plots)
			self.figures = []
		self.results.to_csv(self.save_path, index=False, index_label=False)

	def load(self, path=None):
		path = path or self.save_path
		if os.path.isfile(path):
			self.results.read_csv(path)

	def show(self):
		if len(self.figures) > 0:
			plot = column(*self.figures)
			show(plot)

	def plot(self, message, *kargs, **kwargs):
		p = figure(plot_width=800, plot_height=400,
		           *kargs, **kwargs)
		for mess in message:
			p.line(self.results['round'], self.results[mess], legend_label=mess)
		self.figures.append(p)

	def image(self, *kargs, **kwargs):
		fig = figure()
		fig.image(*kargs, **kwargs)
		self.figures.append(fig)