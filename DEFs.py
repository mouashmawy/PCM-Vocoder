import enum
class Quantizer_types(enum.Enum):
    MID_RISE = 'mid-rise'
    MID_TREAD = 'mid-tread'

class Encoder_types(enum.Enum):
    MANCHESTER = 'manchester'
    ALTERNATE_MARK_INVERSION = 'alternate-mark-inversion'