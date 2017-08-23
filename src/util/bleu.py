#!/usr/bin/env python3

import subprocess


def bleu(where, source, target):
    return subprocess.getoutput(' '.join(
        ['perl', where + '/multi-bleu.perl', '-lc', where + '/' + source, '<', where + '/' + target])
    )
