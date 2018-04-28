#
# device.py, doom-net
#
# Created by Andrey Kolishchak on 04/28/18.
#
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
