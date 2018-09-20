

# def load_binaryproto(mean_file):
#     blob = caffe.proto.caffe_pb2.BlobProto()
#     data = open(mean_file, 'rb').read()
#     blob.ParseFromString(data)
#     arr = np.array(caffe.io.blobproto_to_array(blob))
#     out = arr[0]
#     np.save(npy_file, out)