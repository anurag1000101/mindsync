import hdf5storage  # matlab v5 files
pth='/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/datasets/zuco/task_nr/Matlab files/'
fname='resultsYAC_NR.mat' #'YAC_NR5_EEG.mat'
mat = hdf5storage.loadmat(pth+fname)
print(mat['EEG'])
