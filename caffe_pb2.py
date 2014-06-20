# Generated by the protocol buffer compiler.  DO NOT EDIT!

from google.protobuf import descriptor
from google.protobuf import message
from google.protobuf import reflection
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)



DESCRIPTOR = descriptor.FileDescriptor(
  name='caffe/proto/caffe.proto',
  package='caffe',
  serialized_pb='\n\x17\x63\x61\x66\x66\x65/proto/caffe.proto\x12\x05\x63\x61\x66\x66\x65\"y\n\tBlobProto\x12\x0e\n\x03num\x18\x01 \x01(\x05:\x01\x30\x12\x13\n\x08\x63hannels\x18\x02 \x01(\x05:\x01\x30\x12\x11\n\x06height\x18\x03 \x01(\x05:\x01\x30\x12\x10\n\x05width\x18\x04 \x01(\x05:\x01\x30\x12\x10\n\x04\x64\x61ta\x18\x05 \x03(\x02\x42\x02\x10\x01\x12\x10\n\x04\x64iff\x18\x06 \x03(\x02\x42\x02\x10\x01\"2\n\x0f\x42lobProtoVector\x12\x1f\n\x05\x62lobs\x18\x01 \x03(\x0b\x32\x10.caffe.BlobProto\"i\n\x05\x44\x61tum\x12\x10\n\x08\x63hannels\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\x0c\x12\r\n\x05label\x18\x05 \x01(\x05\x12\x12\n\nfloat_data\x18\x06 \x03(\x02\"|\n\x0f\x46illerParameter\x12\x16\n\x04type\x18\x01 \x01(\t:\x08\x63onstant\x12\x10\n\x05value\x18\x02 \x01(\x02:\x01\x30\x12\x0e\n\x03min\x18\x03 \x01(\x02:\x01\x30\x12\x0e\n\x03max\x18\x04 \x01(\x02:\x01\x31\x12\x0f\n\x04mean\x18\x05 \x01(\x02:\x01\x30\x12\x0e\n\x03std\x18\x06 \x01(\x02:\x01\x31\"\xb3\x07\n\x0eLayerParameter\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x12\n\nnum_output\x18\x03 \x01(\r\x12\x16\n\x08\x62iasterm\x18\x04 \x01(\x08:\x04true\x12-\n\rweight_filler\x18\x05 \x01(\x0b\x32\x16.caffe.FillerParameter\x12+\n\x0b\x62ias_filler\x18\x06 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x0e\n\x03pad\x18\x07 \x01(\r:\x01\x30\x12\x12\n\nkernelsize\x18\x08 \x01(\r\x12\x10\n\x05group\x18\t \x01(\r:\x01\x31\x12\x11\n\x06stride\x18\n \x01(\r:\x01\x31\x12\x33\n\x04pool\x18\x0b \x01(\x0e\x32 .caffe.LayerParameter.PoolMethod:\x03MAX\x12\x1a\n\rdropout_ratio\x18\x0c \x01(\x02:\x03\x30.5\x12\x15\n\nlocal_size\x18\r \x01(\r:\x01\x35\x12\x10\n\x05\x61lpha\x18\x0e \x01(\x02:\x01\x31\x12\x12\n\x04\x62\x65ta\x18\x0f \x01(\x02:\x04\x30.75\x12\x0e\n\x06source\x18\x10 \x01(\t\x12\x10\n\x05scale\x18\x11 \x01(\x02:\x01\x31\x12\x10\n\x08meanfile\x18\x12 \x01(\t\x12\x11\n\tbatchsize\x18\x13 \x01(\r\x12\x13\n\x08\x63ropsize\x18\x14 \x01(\r:\x01\x30\x12\x15\n\x06mirror\x18\x15 \x01(\x08:\x05\x66\x61lse\x12\x1f\n\x05\x62lobs\x18\x32 \x03(\x0b\x32\x10.caffe.BlobProto\x12\x10\n\x08\x62lobs_lr\x18\x33 \x03(\x02\x12\x14\n\x0cweight_decay\x18\x34 \x03(\x02\x12\x14\n\trand_skip\x18\x35 \x01(\r:\x01\x30\x12\x1d\n\x10\x64\x65t_fg_threshold\x18\x36 \x01(\x02:\x03\x30.5\x12\x1d\n\x10\x64\x65t_bg_threshold\x18\x37 \x01(\x02:\x03\x30.5\x12\x1d\n\x0f\x64\x65t_fg_fraction\x18\x38 \x01(\x02:\x04\x30.25\x12\x1a\n\x0f\x64\x65t_context_pad\x18: \x01(\r:\x01\x30\x12\x1b\n\rdet_crop_mode\x18; \x01(\t:\x04warp\x12\x12\n\x07new_num\x18< \x01(\x05:\x01\x30\x12\x17\n\x0cnew_channels\x18= \x01(\x05:\x01\x30\x12\x15\n\nnew_height\x18> \x01(\x05:\x01\x30\x12\x14\n\tnew_width\x18? \x01(\x05:\x01\x30\x12\x1d\n\x0eshuffle_images\x18@ \x01(\x08:\x05\x66\x61lse\x12\x15\n\nconcat_dim\x18\x41 \x01(\r:\x01\x31\".\n\nPoolMethod\x12\x07\n\x03MAX\x10\x00\x12\x07\n\x03\x41VE\x10\x01\x12\x0e\n\nSTOCHASTIC\x10\x02\"T\n\x0fLayerConnection\x12$\n\x05layer\x18\x01 \x01(\x0b\x32\x15.caffe.LayerParameter\x12\x0e\n\x06\x62ottom\x18\x02 \x03(\t\x12\x0b\n\x03top\x18\x03 \x03(\t\"\x85\x01\n\x0cNetParameter\x12\x0c\n\x04name\x18\x01 \x01(\t\x12&\n\x06layers\x18\x02 \x03(\x0b\x32\x16.caffe.LayerConnection\x12\r\n\x05input\x18\x03 \x03(\t\x12\x11\n\tinput_dim\x18\x04 \x03(\x05\x12\x1d\n\x0e\x66orce_backward\x18\x05 \x01(\x08:\x05\x66\x61lse\"\xff\x02\n\x0fSolverParameter\x12\x11\n\ttrain_net\x18\x01 \x01(\t\x12\x10\n\x08test_net\x18\x02 \x01(\t\x12\x14\n\ttest_iter\x18\x03 \x01(\x05:\x01\x30\x12\x18\n\rtest_interval\x18\x04 \x01(\x05:\x01\x30\x12\x0f\n\x07\x62\x61se_lr\x18\x05 \x01(\x02\x12\x0f\n\x07\x64isplay\x18\x06 \x01(\x05\x12\x10\n\x08max_iter\x18\x07 \x01(\x05\x12\x11\n\tlr_policy\x18\x08 \x01(\t\x12\r\n\x05gamma\x18\t \x01(\x02\x12\r\n\x05power\x18\n \x01(\x02\x12\x10\n\x08momentum\x18\x0b \x01(\x02\x12\x14\n\x0cweight_decay\x18\x0c \x01(\x02\x12\x10\n\x08stepsize\x18\r \x01(\x05\x12\x13\n\x08snapshot\x18\x0e \x01(\x05:\x01\x30\x12\x17\n\x0fsnapshot_prefix\x18\x0f \x01(\t\x12\x1c\n\rsnapshot_diff\x18\x10 \x01(\x08:\x05\x66\x61lse\x12\x16\n\x0bsolver_mode\x18\x11 \x01(\x05:\x01\x31\x12\x14\n\tdevice_id\x18\x12 \x01(\x05:\x01\x30\"S\n\x0bSolverState\x12\x0c\n\x04iter\x18\x01 \x01(\x05\x12\x13\n\x0blearned_net\x18\x02 \x01(\t\x12!\n\x07history\x18\x03 \x03(\x0b\x32\x10.caffe.BlobProto')



_LAYERPARAMETER_POOLMETHOD = descriptor.EnumDescriptor(
  name='PoolMethod',
  full_name='caffe.LayerParameter.PoolMethod',
  filename=None,
  file=DESCRIPTOR,
  values=[
    descriptor.EnumValueDescriptor(
      name='MAX', index=0, number=0,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='AVE', index=1, number=1,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='STOCHASTIC', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1344,
  serialized_end=1390,
)


_BLOBPROTO = descriptor.Descriptor(
  name='BlobProto',
  full_name='caffe.BlobProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='num', full_name='caffe.BlobProto.num', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='channels', full_name='caffe.BlobProto.channels', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='height', full_name='caffe.BlobProto.height', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='width', full_name='caffe.BlobProto.width', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='data', full_name='caffe.BlobProto.data', index=4,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=descriptor._ParseOptions(descriptor_pb2.FieldOptions(), '\020\001')),
    descriptor.FieldDescriptor(
      name='diff', full_name='caffe.BlobProto.diff', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=descriptor._ParseOptions(descriptor_pb2.FieldOptions(), '\020\001')),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=34,
  serialized_end=155,
)


_BLOBPROTOVECTOR = descriptor.Descriptor(
  name='BlobProtoVector',
  full_name='caffe.BlobProtoVector',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='blobs', full_name='caffe.BlobProtoVector.blobs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=157,
  serialized_end=207,
)


_DATUM = descriptor.Descriptor(
  name='Datum',
  full_name='caffe.Datum',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='channels', full_name='caffe.Datum.channels', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='height', full_name='caffe.Datum.height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='width', full_name='caffe.Datum.width', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='data', full_name='caffe.Datum.data', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value="",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='label', full_name='caffe.Datum.label', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='float_data', full_name='caffe.Datum.float_data', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=209,
  serialized_end=314,
)


_FILLERPARAMETER = descriptor.Descriptor(
  name='FillerParameter',
  full_name='caffe.FillerParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='type', full_name='caffe.FillerParameter.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=unicode("constant", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='value', full_name='caffe.FillerParameter.value', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='min', full_name='caffe.FillerParameter.min', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='max', full_name='caffe.FillerParameter.max', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='mean', full_name='caffe.FillerParameter.mean', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='std', full_name='caffe.FillerParameter.std', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=316,
  serialized_end=440,
)


_LAYERPARAMETER = descriptor.Descriptor(
  name='LayerParameter',
  full_name='caffe.LayerParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='name', full_name='caffe.LayerParameter.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='type', full_name='caffe.LayerParameter.type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='num_output', full_name='caffe.LayerParameter.num_output', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='biasterm', full_name='caffe.LayerParameter.biasterm', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='weight_filler', full_name='caffe.LayerParameter.weight_filler', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='bias_filler', full_name='caffe.LayerParameter.bias_filler', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='pad', full_name='caffe.LayerParameter.pad', index=6,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='kernelsize', full_name='caffe.LayerParameter.kernelsize', index=7,
      number=8, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='group', full_name='caffe.LayerParameter.group', index=8,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='stride', full_name='caffe.LayerParameter.stride', index=9,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='pool', full_name='caffe.LayerParameter.pool', index=10,
      number=11, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='dropout_ratio', full_name='caffe.LayerParameter.dropout_ratio', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.5,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='local_size', full_name='caffe.LayerParameter.local_size', index=12,
      number=13, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=5,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='alpha', full_name='caffe.LayerParameter.alpha', index=13,
      number=14, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='beta', full_name='caffe.LayerParameter.beta', index=14,
      number=15, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.75,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='source', full_name='caffe.LayerParameter.source', index=15,
      number=16, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='scale', full_name='caffe.LayerParameter.scale', index=16,
      number=17, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='meanfile', full_name='caffe.LayerParameter.meanfile', index=17,
      number=18, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='batchsize', full_name='caffe.LayerParameter.batchsize', index=18,
      number=19, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='cropsize', full_name='caffe.LayerParameter.cropsize', index=19,
      number=20, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='mirror', full_name='caffe.LayerParameter.mirror', index=20,
      number=21, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='blobs', full_name='caffe.LayerParameter.blobs', index=21,
      number=50, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='blobs_lr', full_name='caffe.LayerParameter.blobs_lr', index=22,
      number=51, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='weight_decay', full_name='caffe.LayerParameter.weight_decay', index=23,
      number=52, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='rand_skip', full_name='caffe.LayerParameter.rand_skip', index=24,
      number=53, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='det_fg_threshold', full_name='caffe.LayerParameter.det_fg_threshold', index=25,
      number=54, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.5,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='det_bg_threshold', full_name='caffe.LayerParameter.det_bg_threshold', index=26,
      number=55, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.5,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='det_fg_fraction', full_name='caffe.LayerParameter.det_fg_fraction', index=27,
      number=56, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.25,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='det_context_pad', full_name='caffe.LayerParameter.det_context_pad', index=28,
      number=58, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='det_crop_mode', full_name='caffe.LayerParameter.det_crop_mode', index=29,
      number=59, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=unicode("warp", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='new_num', full_name='caffe.LayerParameter.new_num', index=30,
      number=60, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='new_channels', full_name='caffe.LayerParameter.new_channels', index=31,
      number=61, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='new_height', full_name='caffe.LayerParameter.new_height', index=32,
      number=62, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='new_width', full_name='caffe.LayerParameter.new_width', index=33,
      number=63, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='shuffle_images', full_name='caffe.LayerParameter.shuffle_images', index=34,
      number=64, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='concat_dim', full_name='caffe.LayerParameter.concat_dim', index=35,
      number=65, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _LAYERPARAMETER_POOLMETHOD,
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=443,
  serialized_end=1390,
)


_LAYERCONNECTION = descriptor.Descriptor(
  name='LayerConnection',
  full_name='caffe.LayerConnection',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='layer', full_name='caffe.LayerConnection.layer', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='bottom', full_name='caffe.LayerConnection.bottom', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='top', full_name='caffe.LayerConnection.top', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=1392,
  serialized_end=1476,
)


_NETPARAMETER = descriptor.Descriptor(
  name='NetParameter',
  full_name='caffe.NetParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='name', full_name='caffe.NetParameter.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='layers', full_name='caffe.NetParameter.layers', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='input', full_name='caffe.NetParameter.input', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='input_dim', full_name='caffe.NetParameter.input_dim', index=3,
      number=4, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='force_backward', full_name='caffe.NetParameter.force_backward', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=1479,
  serialized_end=1612,
)


_SOLVERPARAMETER = descriptor.Descriptor(
  name='SolverParameter',
  full_name='caffe.SolverParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='train_net', full_name='caffe.SolverParameter.train_net', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='test_net', full_name='caffe.SolverParameter.test_net', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='test_iter', full_name='caffe.SolverParameter.test_iter', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='test_interval', full_name='caffe.SolverParameter.test_interval', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='base_lr', full_name='caffe.SolverParameter.base_lr', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='display', full_name='caffe.SolverParameter.display', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='max_iter', full_name='caffe.SolverParameter.max_iter', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='lr_policy', full_name='caffe.SolverParameter.lr_policy', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='gamma', full_name='caffe.SolverParameter.gamma', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='power', full_name='caffe.SolverParameter.power', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='momentum', full_name='caffe.SolverParameter.momentum', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='weight_decay', full_name='caffe.SolverParameter.weight_decay', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='stepsize', full_name='caffe.SolverParameter.stepsize', index=12,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='snapshot', full_name='caffe.SolverParameter.snapshot', index=13,
      number=14, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='snapshot_prefix', full_name='caffe.SolverParameter.snapshot_prefix', index=14,
      number=15, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='snapshot_diff', full_name='caffe.SolverParameter.snapshot_diff', index=15,
      number=16, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='solver_mode', full_name='caffe.SolverParameter.solver_mode', index=16,
      number=17, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='device_id', full_name='caffe.SolverParameter.device_id', index=17,
      number=18, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=1615,
  serialized_end=1998,
)


_SOLVERSTATE = descriptor.Descriptor(
  name='SolverState',
  full_name='caffe.SolverState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='iter', full_name='caffe.SolverState.iter', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='learned_net', full_name='caffe.SolverState.learned_net', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='history', full_name='caffe.SolverState.history', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=2000,
  serialized_end=2083,
)

_BLOBPROTOVECTOR.fields_by_name['blobs'].message_type = _BLOBPROTO
_LAYERPARAMETER.fields_by_name['weight_filler'].message_type = _FILLERPARAMETER
_LAYERPARAMETER.fields_by_name['bias_filler'].message_type = _FILLERPARAMETER
_LAYERPARAMETER.fields_by_name['pool'].enum_type = _LAYERPARAMETER_POOLMETHOD
_LAYERPARAMETER.fields_by_name['blobs'].message_type = _BLOBPROTO
_LAYERPARAMETER_POOLMETHOD.containing_type = _LAYERPARAMETER;
_LAYERCONNECTION.fields_by_name['layer'].message_type = _LAYERPARAMETER
_NETPARAMETER.fields_by_name['layers'].message_type = _LAYERCONNECTION
_SOLVERSTATE.fields_by_name['history'].message_type = _BLOBPROTO
DESCRIPTOR.message_types_by_name['BlobProto'] = _BLOBPROTO
DESCRIPTOR.message_types_by_name['BlobProtoVector'] = _BLOBPROTOVECTOR
DESCRIPTOR.message_types_by_name['Datum'] = _DATUM
DESCRIPTOR.message_types_by_name['FillerParameter'] = _FILLERPARAMETER
DESCRIPTOR.message_types_by_name['LayerParameter'] = _LAYERPARAMETER
DESCRIPTOR.message_types_by_name['LayerConnection'] = _LAYERCONNECTION
DESCRIPTOR.message_types_by_name['NetParameter'] = _NETPARAMETER
DESCRIPTOR.message_types_by_name['SolverParameter'] = _SOLVERPARAMETER
DESCRIPTOR.message_types_by_name['SolverState'] = _SOLVERSTATE

class BlobProto(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _BLOBPROTO
  
  # @@protoc_insertion_point(class_scope:caffe.BlobProto)

class BlobProtoVector(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _BLOBPROTOVECTOR
  
  # @@protoc_insertion_point(class_scope:caffe.BlobProtoVector)

class Datum(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _DATUM
  
  # @@protoc_insertion_point(class_scope:caffe.Datum)

class FillerParameter(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _FILLERPARAMETER
  
  # @@protoc_insertion_point(class_scope:caffe.FillerParameter)

class LayerParameter(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _LAYERPARAMETER
  
  # @@protoc_insertion_point(class_scope:caffe.LayerParameter)

class LayerConnection(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _LAYERCONNECTION
  
  # @@protoc_insertion_point(class_scope:caffe.LayerConnection)

class NetParameter(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _NETPARAMETER
  
  # @@protoc_insertion_point(class_scope:caffe.NetParameter)

class SolverParameter(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _SOLVERPARAMETER
  
  # @@protoc_insertion_point(class_scope:caffe.SolverParameter)

class SolverState(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _SOLVERSTATE
  
  # @@protoc_insertion_point(class_scope:caffe.SolverState)

# @@protoc_insertion_point(module_scope)
