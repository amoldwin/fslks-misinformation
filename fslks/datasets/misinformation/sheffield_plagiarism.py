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

"""Sheffield Plagiarism Dataset."""
import csv
import os
import pandas as pd

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@article{article,
author = {Clough, Paul and Stevenson, Mark},
year = {2011},
month = {03},
pages = {5-24},
title = {Developing a corpus of plagiarised short answers},
volume = {45},
journal = {Language Resources and Evaluation},
doi = {10.1007/s10579-009-9112-1}
}"""



_DESCRIPTION = """
 A corpus consisting of answers to short questions in which plagiarism has been simulated.
"""

_ORIGINAL = "original"
_STUDENT = "student"
_LABEL = "label"

_DOWNLOAD_URL = "https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html"

_CSV_PATH = '../tensorflow_datasets/tensorflow_datasets/downloads/plagiarism/sheffield_plagiarism.csv'
_SHEFFIELDPLAGIARISM_DOWNLOAD_INSTRUCTIONS = """Temp until upload"""


class SheffieldPlagiarism(tfds.core.GeneratorBasedBuilder):
    """Contradictions between Medline abstracts and questions."""
    VERSION = tfds.core.Version("1.0.0")
    MANUAL_DOWNLOAD_INSTRUCTIONS = _SHEFFIELDPLAGIARISM_DOWNLOAD_INSTRUCTIONS

    def _info(self):

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                _ORIGINAL: tfds.features.Text(),
                _STUDENT: tfds.features.Text(),
                _LABEL: tfds.features.ClassLabel(num_classes=5),
            }),
            supervised_keys=None,
            homepage="https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html",
            citation=_CITATION,
        )
    
            

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        path = os.path.join(dl_manager.manual_dir, self.name)
   
        return [
        tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"path": os.path.join(path,  "sheffield_plagiarism_train.csv")},
            ),
        tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={"path": os.path.join(path,  "sheffield_plagiarism_test.csv")},
            ),
        ]

    def _generate_examples(self, path=_CSV_PATH):
        """Yields examples."""
        # Header fo csv: REVIEW_PMID,REVIEW_TITLE,ASSERTION,PMID,QUESTION,TYPE,TEXT,length,abstracts   
        df = pd.read_csv(path)
        # with tf.io.gfile.GFile(path) as f:
        for i, row in df.iterrows():
            yield str(i + 1), {
                _ORIGINAL: str(row['original']),
                _STUDENT: str(row['student']),
                _LABEL: str(row['label'])
            }

