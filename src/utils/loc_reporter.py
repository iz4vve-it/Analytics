#!/usr/bin/env python
# coding: utf-8

__author__ = "Pietro Mascolo"
__version__ = "1.0"
__copyright__ = "Copyright 2015, Pietro Mascolo"
__email__ = "pietro@mascolo.eu"
__status__ = "Prototype"

"""
Modulo per il conto delle righe di un file sorgente.

Allo stato attuale di implementazione e' possibile
escludere o includere le righe commentate o vuote.

Utilizzo:

loc_reporter [-f] [estensione_file] [-c] [-e]

- ./loc_reporter.py -f py -c
    conto LoC di un file python includendo le righe commentate, ma escludendo quelle vuote

- ./loc_reporter.py -f c
    conto LoC di un file C escludendo le righe vuote o commentate

- ./loc_reporter.py -f java -c -e
    conto LoC di un file java includendo sia righe commentate, sia righe vuote

- ./loc_reporter.py -f rb -c -e --exclude "main, socket"
    conto LoC di un file ruby includendo sia righe commentate, sia righe vuote e
    non considerando file con "main" o "socket" nel nome o nel percorso
"""

import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="file extension")
parser.add_argument("-c", help="exclude commented lines", action="store_true")
parser.add_argument("-e", help="exclude empty lines", action="store_true")
parser.add_argument("--exclude", help="names to be excluded")
parser.add_argument("--folder", help="folder to scan")
parser.add_argument("-edi", action="store_true")
args = parser.parse_args()

FILE_FORMAT = args.f
COMMENTED_LINES = args.c
EMPTY_LINES = args.e
EXCLUDE = args.exclude
EDI = args.edi
FOLDER = args.folder

REGEX_COMMENTED_LINE = re.compile("^(\s*|\t*)#")
REGEX_EMPTY_LINE = re.compile("^(\s*|\t*)$")

COMMENT_CHARACTER = {
    "py": "#",
    "pl": "#",
    "rb": "#",
    "c": "//",
    "cpp": "//",
    "cs": "//",
    "java": "//",
    "js": "//",
    "html": "",
    "f": "!",
    "for": "!",
    "f90": "!",
    "f95": "!",
    "m": "%",
}

length = 65
length_title = length - 2


def count_lines(exclude=""):
    """
    :param folder: cartella in cui effettuare il walk
    :param exclude: file e percorsi da escludere
    :return: None
    """
    loc_dict = {}
    os.system("clear")
    file_list = []

    # se ci sono file da escludere li inserisco qua
    if EXCLUDE:
        exclude = EXCLUDE.replace(" ", "").split(",")

    folder = FOLDER or "."

    # walk su tutta la struttura della cartella selezionata
    for root, dirs, filenames in os.walk(folder):
        if filenames:
            for name in filenames:
                if name.endswith(".{}".format(FILE_FORMAT)):
                    path = os.path.join(root, name)
                    checks = [item in path for item in exclude]  # check per escludere i file nella lista exclude
                    if not any(checks):
                        file_list.append(os.path.join(root, name))

    for filename in file_list:
        with open(filename) as f:
            lines = f.read()

        if COMMENTED_LINES and EMPTY_LINES:
            # conta tutte le righe del file
            key = "{:<40}".format(filename)
            loc_dict[key] = sum(1 for line in lines.split("\n"))

        elif not COMMENTED_LINES:
            # esclude le righe commentate: quelle che iniziano col carattere definito in COMMENT_CHARACTER
            # TODO implementare l'esclusione dei commenti multiriga

            key = "{:<40}".format(filename)
            loc_dict[key] = sum(1 for line in lines.split("\n")
                                if not line.strip().startswith("{}".format(COMMENT_CHARACTER[FILE_FORMAT])))

        elif not EMPTY_LINES:
            # esclude le righe vuote

            key = "{:<40}".format(filename)
            loc_dict[key] = sum(1 for line in lines.split("\n") if line.strip())

        else:
            # esclude le righe vuote o commentate

            key = "{:<40}".format(filename)
            loc_dict[key] = sum(1 for line in lines.split("\n")
                                if line.strip() and
                                not line.strip().startswith("{}".format(COMMENT_CHARACTER[FILE_FORMAT])))

    string_to_print = "Total LoC: {}".format(sum(loc_dict.values()))
    print '\n\n'
    print "#" * length
    print "#{:^63}#".format(string_to_print)
    print '#' * length
    sorted_loc_dict = sorted(loc_dict.items(), key=lambda x: x[1])

    print "\nFiles by lines of code:\n"
    for item in sorted_loc_dict[::-1]:
        print "".join([c for c in str(item).replace(",", "\t=>\t") if c not in ("(", ")", "[", "]", "'")])
    print '\n\n'


if __name__ == '__main__':
    # TODO implementare un arg per passare un percorso da riga di comando per la directory da scansionare
    count_lines()
