This archive contains the evaluation script for the Semdis 2014
lexical substitution task for French. More information about the
evaluation measures can be found on our website [1].

The archive consists of the following files:

* semdis_eval.py - python script to compute the evaluation measures;

* mapping.txt - mapping of participants' propositions to canonical
  forms. More information can be found in the file's comment header;

* semdis2014_lexsub_gold.txt - gold standard annotations for the
  lexical substitution task;

* baseline_dicosyn.txt - baseline for the lexical substitution task,
  created by taking all the synonyms for a particular item from the
  lexical resource DICOSYN (sorted according to frequency in the FRWaC
  corpus).

The evaluation script's usage is the following:

python semdis_eval.py -g GOLDFILE -t TESTFILE

e.g. python semdis_eval.py -g semdis2014_lexsub_gold.txt -t baseline_dicosyn.txt

Optionally, a particular measure can be defined, and the measures can
be computed without the normalization mapping. More information is
provided by typing

python semdis_eval.py -h

The Semdis 2014 lexical substitution task for French is licensed under a Creative Commons
Attribution-NonCommercial-ShareAlike 3.0 Unported License [2].

[1] http://www.irit.fr/semdis2014/
[2] http://creativecommons.org/licenses/by-nc-sa/3.0/
