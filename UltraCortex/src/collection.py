import time
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams


'''
Collecting raw EEG with Ultracortex headset
Ubuntu serial port --> '/dev/ttyUSB0' (https://docs.openbci.com/GettingStarted/Boards/CytonGS/)
Cyton Board ID--> 0 (8 channels)
'''
parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str, 
        help='serial port', required=False, default='/dev/ttyUSB0')
parser.add_argument('--board-id', type=int, 
        help='board id, check docs to get a list of supported boards', required=False, default=0)
args = parser.parse_args()

'''
Brainflow API: BrainFlowInputParams (parameter config)
Brainflow API: BoardShim (Access data stream from board)
'''
print('[*] Adding board settings to input config ...')
params = BrainFlowInputParams()
params.serial_port = args.serial_port
board = BoardShim(args.board_id, params)

'''
Data collection: time --> 3 seconds (line 41 blocking call)
'''
BoardShim.enable_dev_board_logger()
print('[*] Prepairing the session ...')
board.prepare_session()
print('[*] Starting Data Stream ...')
board.start_stream()
time.sleep(3)
print('[*] Loading data ...')
data = board.get_board_data(500)
print('[*] Stop stream and release resources ...')
board.stop_stream()
board.release_session()

'''
Data extraction: 24 channels in total--EEG (8 channels)
'''
eeg_channels = BoardShim.get_eeg_channels(args.board_id)
eeg_names = BoardShim.get_eeg_names(args.board_id)

df = pd.DataFrame(np.transpose(data[:,1:]))
df_eeg = df[eeg_channels]
df_eeg.columns = eeg_names
df_eeg.plot(subplots=True, sharex=True, legend=True)
plt.legend(loc='lower right')
plt.show()

'''
Storing CSV with DatafRame and headers
'''
timestr = time.strftime("%Y%m%d-%H%M%S")
filename = timestr + '.csv'
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'UltraCortex/data', filename)
df_eeg.to_csv(data_dir, sep=',', index = False)
