import argparse
import logging

import pyqtgraph as pg
#from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.Qt import QtGui, QtWidgets, QtCore

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowFunctions

import random
import time
import requests

from pythonosc import udp_client
import numpy as np

### only for sending fake data
import pandas as pd

counter = 0

port = 5005
ip = '127.0.0.1'
client = udp_client.SimpleUDPClient(ip, port)

electrode_num = [1,2,7,8]

def prepData(d1, ch, sampling_rate): #notch and bandpass filters
    DataFilter.detrend(d1[ch], DetrendOperations.CONSTANT.value)
    DataFilter.perform_highpass(d1[ch], sampling_rate, 2, 1, 0, 1) # removes DC offset
    DataFilter.perform_bandpass(d1[ch], sampling_rate, 1.0, 40.0, 2, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_bandstop(d1[ch], sampling_rate, 50.0, 60.0, 3, FilterTypes.BUTTERWORTH.value, 0)

def filterBank_1(d1, ch, sampling_rate,nfft):
    
    center_freq1 = 12.04 #from arduino measurement)
    
    the_nfft = int(nfft*2.0)
    psd_bg = DataFilter.get_psd_welch(d1[ch], the_nfft, the_nfft//2, sampling_rate, 1)
    band_power_bg = (DataFilter.get_band_power(psd_bg, 7.0, 27.0))/20.0
    
    d_copy = np.copy(d1)
    DataFilter.perform_bandpass(d_copy[ch], sampling_rate, center_freq1-1.0, center_freq1+1.0, 4, FilterTypes.CHEBYSHEV_TYPE_1.value, 1.0)
    psd = DataFilter.get_psd_welch(d_copy[ch], the_nfft, the_nfft//2, sampling_rate, 1)
    band_power = DataFilter.get_band_power(psd, center_freq1-0.5, center_freq1+0.5)
    
    #print(band_power, band_power_bg)
    return band_power, band_power_bg
    
def filterBank_2(d1, ch, sampling_rate,nfft):
    center_freq = 15.6 #from arduino measurement)
    
    the_nfft = int(nfft*2.0)
    psd_bg = DataFilter.get_psd_welch(d1[ch], the_nfft, the_nfft//2, sampling_rate, 1)
    band_power_bg = (DataFilter.get_band_power(psd_bg, 7.0, 27.0))/20.0
    
    d_copy2 = np.copy(d1)
    DataFilter.perform_bandpass(d_copy2[ch], sampling_rate, center_freq-1.0, center_freq+1.0, 4, FilterTypes.CHEBYSHEV_TYPE_1.value, 1.0)
    psd = DataFilter.get_psd_welch(d_copy2[ch], the_nfft, the_nfft//2, sampling_rate, 1)
    band_power = DataFilter.get_band_power(psd, center_freq-0.5, center_freq+0.5)
    
    #print(band_power, band_power_bg)
    return band_power, band_power_bg
    

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 5000 #this is also the size of data array in channel
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        self.nfft = DataFilter.get_nearest_power_of_two(self.sampling_rate)

        self.app = QtWidgets.QApplication([])
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        
        QtWidgets.QApplication.instance().exec_()

    def update(self):
 
        data = self.board_shim.get_current_board_data(self.num_points) ### uncomment here for real data

        for i in electrode_num:
            prepData(data,i, self.sampling_rate)
        
        bp_1, g1 = filterBank_1(data, 1, self.sampling_rate, self.nfft)
        bp_2, g2 = filterBank_1(data, 2, self.sampling_rate, self.nfft)
        bp_7, g7 = filterBank_1(data, 7, self.sampling_rate, self.nfft) # occipital
        avg_bg = bp_7/(float(bp_1+bp_2)/2.0)
        
        bp_12, g12 = filterBank_2(data, 1, self.sampling_rate, self.nfft)
        bp_22, g22 = filterBank_2(data, 2, self.sampling_rate, self.nfft)
        bp_72, g72 = filterBank_2(data, 7, self.sampling_rate, self.nfft) # occipital
        avg_bg2 = bp_72/(float(bp_12+bp_22)/2.0)
                
        print("%.2f" % bp_7, "%.2f" % avg_bg, "%.2f" % g7)
        
        ## positive detection defined by:
        # bp_7/g7 > 2.0  (SNR of electrode 7 to its own noise)
        # electrode signal shold be less than 10 (optional?)
        # bp_7/g7 divided by avg_bg > 0.7

        # also make sure to map the ratios and values
        tot_vol = bp_7
        tot_vol2 = bp_72
        
        ## ensure no division by 0:
        if (g7>0 and bp_7>0):
            signal_ratio = bp_7/g7
        else:
            signal_ratio = 0
            
        if (g72>0 and bp_72>0):
            signal_ratio2 = bp_72/g72
        else:
            signal_ratio2 = 0

        ## scaling of signals:
        if signal_ratio > 2 and signal_ratio < 6.0: # gongs
            signal_ratio = signal_ratio/6.0
        elif signal_ratio >= 6.0:
            signal_ratio = 1.0
        else:
            signal_ratio = 0.1
        
        ## scaling of signals:
        if signal_ratio2 > 2 and signal_ratio2 < 6.0: # bowls
            signal_ratio2 = signal_ratio2/6.0
        elif signal_ratio2 >= 6.0:
            signal_ratio2 = 1.0
        else:
            signal_ratio2 = 0.1
        
        #if tot_vol > 3 and tot_vol < 10.0:
        #    tot_vol = tot_vol/10.0
        #elif tot_vol >= 10.0:
        #    tot_vol = 1.0
        #else:
        #    tot_vol = 0.3

        #hz12 = [tot_vol, signal_ratio]
        
        # signal_ratio: 12Hz --> gongs --> short LED
        # signal_ratio2: 15.5 Hz --> bowls --> long LED
        hz12 = [signal_ratio, signal_ratio2]
        print(hz12)

        ### send OSC message with data array
        client.send_message("/ch8", hz12)
        
        self.app.processEvents()

def main():

    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='/dev/cu.usbserial-DM03H5VO')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.CYTON_BOARD.value)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    args = parser.parse_args()
    
    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    try:
        board_shim = BoardShim(args.board_id, params)
        board_shim.prepare_session() ### uncomment here for real data
        board_shim.start_stream(450000, args.streamer_params) ### uncomment here for real data
        Graph(board_shim)
        
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()
            
    if KeyboardInterrupt():
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()
            exit()

if __name__ == '__main__':
    main()
