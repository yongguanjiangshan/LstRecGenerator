from gluoncv.data import RecordFileDetection
record_dataset = RecordFileDetection('voc2007.rec', coord_normalized=True)

# we expect same results from LstDetection
print('length:', len(record_dataset))
first_img = record_dataset[3][0]
print('image shape:', first_img.shape)
print('Label example:')
print(record_dataset[3][1])

record_dataset = RecordFileDetection('voc2007.rec')
img, label = record_dataset[6]
print('imgShape: '+str(img.shape), 'labelShape: '+str(label.shape))