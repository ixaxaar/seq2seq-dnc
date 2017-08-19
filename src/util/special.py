#!/usr/bin/env python3

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
SEP_token = 4
RESERVED1_token = 5
RESERVED2_token = 6

SOS = SOS_token
EOS = EOS_token
PAD = PAD_token
UNK = UNK_token
SEP = SEP_token

default_dict = {
    0: "PAD",
    1: "SOS",
    2: "EOS",
    3: "UNK",
    4: "SEP",
    5: "RESERVED1",
    6: "RESERVED2"
}

MAX_LENGTH = 100
