# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Big Bench Hard Dataset"""

import json
import datasets

all_configs = [
    'salient_translation_error_detection',
    'tracking_shuffled_objects_three_objects',
    'geometric_shapes',
    'object_counting',
    'word_sorting',
    'logical_deduction_five_objects',
    'hyperbaton',
    'sports_understanding',
    'logical_deduction_seven_objects',
    'multistep_arithmetic_two',
    'ruin_names',
    'causal_judgement',
    'logical_deduction_three_objects',
    'formal_fallacies',
    'snarks',
    'boolean_expressions',
    'reasoning_about_colored_objects',
    'dyck_languages',
    'navigate',
    'disambiguation_qa',
    'temporal_sequences',
    'web_of_lies',
    'tracking_shuffled_objects_five_objects',
    'penguins_in_a_table',
    'movie_recommendation',
    'date_understanding'
    ]

_CITATION = """\
@article{suzgun2022challenging,
  title={Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them},
  author={Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V and Chi, Ed H and Zhou, Denny and and Wei, Jason},
  journal={arXiv preprint arXiv:2210.09261},
  year={2022}
}
"""

_DESCRIPTION = """\
BIG-Bench (Srivastava et al., 2022) is a diverse evaluation suite that focuses on tasks believed to be beyond the capabilities of current language models. Language models have already made good progress on this benchmark, with the best model in the BIG-Bench paper outperforming average reported human-rater results on 65% of the BIG-Bench tasks via few-shot prompting. But on what tasks do language models fall short of average human-rater performance, and are those tasks actually unsolvable by current language models?
"""

_HOMEPAGE = "https://github.com/suzgunmirac/BIG-Bench-Hard"

_LICENSE = "MIT license"

_URLS = {
    config: f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{config}.json" for config in all_configs
}


class BigBenchHard(datasets.GeneratorBasedBuilder):
    """A subset of Big Bench Dataset that is found to be particularly hard for current generation of LLMs"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=config, description=f"This subset of BBH dataset covers {config}") for config in all_configs
    ]

    def _info(self):
        features = datasets.Features(
                {
                    "input": datasets.Value("string"),
                    "target": datasets.Value("string"),
                }
            )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features, 
            supervised_keys=("input", "target"),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]
        data_file = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_file,
                    "split": "train",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)['examples']
            for key, row in enumerate(data):
                yield key, {
                    "input": row["input"],
                    "target": row["target"],
                }