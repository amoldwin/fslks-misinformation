# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BioContradiction Dataset."""
import csv
import os

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@Article{Alamri2016,
author={Alamri, Abdulaziz
and Stevenson, Mark},
title={A corpus of potentially contradictory research claims from cardiovascular research abstracts},
journal={Journal of Biomedical Semantics},
year={2016},
month={Jun},
day={07},
volume={7},
number={1},
pages={36},
abstract={Research literature in biomedicine and related fields contains a huge number of claims, such 
as the effectiveness of treatments. These claims are not always consistent and may even contradict each 
other. Being able to identify contradictory claims is important for those who rely on the biomedical literature. 
Automated methods to identify and resolve them are required to cope with the amount of information available. 
However, research in this area has been hampered by a lack of suitable resources. We describe a methodology to 
develop a corpus which addresses this gap by providing examples of potentially contradictory claims and demonstrate  
how it can be applied to identify these claims from Medline abstracts related to the topic of cardiovascular disease.},
issn={2041-1480},
doi={10.1186/s13326-016-0083-z},
url={https://doi.org/10.1186/s13326-016-0083-z}
}"""



_DESCRIPTION = """
A set of systematic reviews concerned with four topics in cardiovascular disease were identified from Medline and analysed
 to determine whether the abstracts they reviewed contained contradictory research claims. For each review, annotators were
  asked to analyse these abstracts to identify claims within them that answered the question addressed in the review. The 
  annotators were also asked to indicate how the claim related to that question and the type of the claim.
"""

_REVIEW_TITLE = "title"
_QUESTION = "question"
_ABSTRACT_TEXT = "text"
_ABSTRACT = "abstract"
_LABEL = "label"
_BEST_SENTENCE = 'best_sentence'

_DOWNLOAD_URL = "http://staffwww.dcs.shef.ac.uk/people/M.Stevenson/resources/bio_contradictions/corpus.xml"

_CSV_PATH = '/data/LHC_kitchensink/data/bio_contradictions/question_abstract.csv '
_BIOCONTRADICTION_DOWNLOAD_INSTRUCTIONS = """Temp until upload"""


class RankPredict(tfds.core.GeneratorBasedBuilder):
    """Contradictions between Medline abstracts and questions."""
    VERSION = tfds.core.Version("1.0.0")
    MANUAL_DOWNLOAD_INSTRUCTIONS = _BIOCONTRADICTION_DOWNLOAD_INSTRUCTIONS

    def _info(self):

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                _REVIEW_TITLE: tfds.features.Text(),
                _QUESTION: tfds.features.Text(),
                _ABSTRACT_TEXT: tfds.features.Text(),
                _REVIEW_TITLE: tfds.features.Text(),
                _BEST_SENTENCE: tfds.features.Text(),
                _LABEL: tfds.features.ClassLabel(num_classes=2),
            }),
            supervised_keys=None,
            homepage="http://staffwww.dcs.shef.ac.uk/people/M.Stevenson/resources/bio_contradictions/",
            citation=_CITATION,
        )
    
            

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        path = os.path.join(dl_manager.manual_dir, 'biocontradiction')
        # assert False, path
        # dl_dir = dl_manager.download_and_extract(_DOWNLOAD_URL)
        # dl_dir = os.path.join(dl_dir, r'biocontradiction')
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"path": os.path.join(path, "biocontradiction.csv")},
            )
        ]

    def _generate_examples(self, path=_CSV_PATH):
        """Yields examples."""
        # Header fo csv: REVIEW_PMID,REVIEW_TITLE,ASSERTION,PMID,QUESTION,TYPE,TEXT,length,abstracts   
        with tf.io.gfile.GFile(path) as f:
            for i, row in enumerate(csv.DictReader(f)):
                yield str(i + 1), {
                    _QUESTION: row['QUESTION'],
                    _REVIEW_TITLE: row['REVIEW_TITLE'],
                    _ABSTRACT_TEXT: row['abstracts'],
                    _BEST_SENTENCE:row['TEXT'],
                    _LABEL: 0 if row['ASSERTION']=='YS' else 1 if row['ASSERTION']=='NO' else -1 
                }
