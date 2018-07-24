import tensorflow as tf
from model import Model
from bilinear_diag import BilinearDiag

class MergeLayer(BilinearDiag):
	''' The layer where linkpd and nodecf merge into. '''

	def __init__(self, next_component, settings):
		if len(next_component) == 2:
			self.task1 = next_component[1]
			self.task2 = next_component[2]
		else:
        	self.task1 = next_component
        	self.task2 = None
        #print('next_component',next_component)
        self.settings = settings

        self.entity_count = int(self.settings['EntityCount'])
        self.relation_count = int(self.settings['RelationCount'])
        self.edge_count = int(self.settings['EdgeCount'])

        self.parse_settings()

	def get_loss(self, mode='train'):
		if self.task1 is not None:
			self.loss = self.task1.get_loss(mode)
			if self.task2 is not None:
				self.loss += self.task2.get_loss(mode)
			return self.loss
		raise NotImplementedError

	def local_get_regularization(self):
		if self.task1 is not None:
			self.regularizaiton = self.task1.local_get_regularization()
			if self.task2 is not None:
				self.regularizaiton += self.task2.local_get_regularization()
			return self.regularizaiton
		raise NotImplementedError
