import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

class LSTM(L.LightningModule):
	def __init__(self): # Create and initialise weight and bias tensors
		super().__init__()
		mean = torch.tensor(0.0)
		std = torch.tensor(1.0)

		self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
		self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
		self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

		self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
		self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
		self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

		self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
		self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
		self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

		self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
		self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
		self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

	def lstm_unit(self, input_value, long_memory, short_memory): # Do the LSTM math
		# See https://higanbana1516.s-ul.eu/5bpd1IFZ for calculation diagram

		# The first stage calculates the percentage of the long-term memory to remember
		long_remember_percent = torch.sigmoid((short_memory * self.wlr1) + (input_value * self.wlr2) + self.blr1)

		# The second stage creates a new potential long-term memory and determines what percentage of it to remember
		potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) + (input_value * self.wpr2) + self.bpr1)
		potential_memory = torch.tanh((short_memory * self.wp1) + (input_value * self.wp2) + self.bp1)

		# Then we update the long-term memory
		updated_long_memory = ((long_memory * long_remember_percent) + (potential_remember_percent * potential_memory))
		# In the third stage, we create a new short-term memory and determine what percentage to remember
		output_percent = torch.sigmoid((short_memory * self.wo1) + (input_value * self.wo2) + self.bo1)
		updated_short_memory = torch.tanh(updated_long_memory) * output_percent

		# Lastly, we return the updated long and short-term memories
		return ([updated_long_memory, updated_short_memory])

	def forward(self, input): # Make a forward path through LSTM
		long_memory = 0
		short_memory = 0

		for i in input:
			long_memory, short_memory = self.lstm_unit(i, long_memory, short_memory)

		return short_memory

	def configure_optimizers(self): # Configure Adam optimizer
		return Adam(self.parameters())

	def training_step(self, batch, batch_idx): # Calculate loss and log training process
		input_i, label_i = batch
		output_i = self.forward(input_i[0])
		loss = (output_i - label_i) ** 2

		self.log("train_loss", loss)

		if (label_i == 0):
			self.log("out_0", output_i)
		else:
			self.log("out_1", output_i)

		return loss
