import csv
import re
import ast

class PromptType():
    def __init__(self,language = "en"):
        if language == "en":
            self.language = "en"
        elif language == "bn":
            self.language = "bn"
        else:
            raise ValueError("Language not supported")
        
        if self.language == "en":
            self.process_funcs = {
                    'openbookqa': self.process_openbookqa,
                    'arc-easy': self.process_arc,
                    'arc-challenge': self.process_arc,
                    'truthfulqa-mc': self.process_truthfulqa_mc,
                    'truthfulqa-ml': self.process_truthfulqa_ml,
                    'bbh-date': self.process_bbh_date,
                    'bbh-disamb-qa': self.process_bbh_general,
                    'bbh-geo-shapes':self.process_bbh_general,
                    'bbh-hyperbaton': self.process_bbh_general,
                    'bbh-logic-3': self.process_bbh_general,
                    'bbh-logic-5': self.process_bbh_general,
                    'bbh-logic-7': self.process_bbh_general,
                    'bbh-movie': self.process_bbh_general,
                    'bbh-penguins': self.process_bbh_general,
                    'bbh-reasoning': self.process_bbh_general,
                    'bbh-ruin-names': self.process_bbh_general,
                    'bbh-salient': self.process_bbh_general,
                    'bbh-snarks': self.process_bbh_general,
                    'bbh-temporal': self.process_bbh_general,
                    'bbh-track-3': self.process_bbh_general,
                    'bbh-track-5': self.process_bbh_general,
                    'bbh-track-7': self.process_bbh_general,
                    'commonsenseqa': self.process_commonsenseqa,
                    'piqa': self.process_piqa,
                    'mmlu': self.process_mmlu,
                    'gsm8k-main': self.process_gsm8k,
                    'winogrande': self.process_winogrande,
                    'boolq': self.process_boolq,
                    'hellaswag':self.process_hellaswag,
                    'titullm': self.process_titullm
                    # ... add more
                    }  
            
            # initilizing with default system_msg, overwrite if required
            self.sys_msg = {key: self.sys_msg_general() for key in self.process_funcs.keys()}
            # overwrite example
            self.sys_msg['gsm8k-main'] = self.sys_msg_gsm8k()
            self.sys_msg['boolq'] = self.sys_msg_boolq()
                            
            self.inp_msg = {
                            'openbookqa': self.input_msg_mcq(4),
                            'arc-easy': self.input_msg_mcq(5),
                            'arc-challenge': self.input_msg_mcq(5),
                            'truthfulqa-mc': self.input_msg_mcq(13),
                            'truthfulqa-ml': self.input_msg_truthfulqa_ml,
                            'bbh-date': self.input_msg_mcq(4),
                            'bbh-disamb-qa': self.input_msg_mcq(4),
                            'bbh-geo-shapes':self.input_msg_mcq(4),
                            'bbh-hyperbaton': self.input_msg_mcq(4),
                            'bbh-logic-3': self.input_msg_mcq(4),
                            'bbh-logic-5': self.input_msg_mcq(4),
                            'bbh-logic-7': self.input_msg_mcq(4),
                            'bbh-movie': self.input_msg_mcq(4),
                            'bbh-penguins': self.input_msg_mcq(4),
                            'bbh-reasoning': self.input_msg_mcq(4),
                            'bbh-ruin-names': self.input_msg_mcq(4),
                            'bbh-salient': self.input_msg_mcq(4),
                            'bbh-snarks': self.input_msg_mcq(4),
                            'bbh-temporal': self.input_msg_mcq(4),
                            'bbh-track-3': self.input_msg_mcq(4),
                            'bbh-track-5': self.input_msg_mcq(4),
                            'bbh-track-7': self.input_msg_mcq(4),
                            'commonsenseqa': self.input_msg_mcq(5),
                            'piqa': self.input_msg_mcq(2),
                            'mmlu': self.input_msg_mcq(4),
                            'gsm8k-main': self.input_msg_gsm8k(),
                            'winogrande': self.input_msg_mcq(2),
                            'boolq': self.input_msg_boolq(),
                            'hellaswag':self.input_msg_mcq(4),
                            'titullm': self.input_msg_mcq(4)
                            # ... add more
                            }  
            # self.acc_func = {
            #     'openbookqa': self.acc_func_openbookqa,
            #     'arc-easy': self.acc_func_arc,
            #     'arc-challenge': self.acc_func_arc,
            #     'truthfulqa-mc': self.acc_func_truthfulqa_mc,
            #     'truthfulqa-ml': self.acc_func_truthfulqa_ml,
            #     'commonsenseqa': self.acc_func_commonsenseqa,
            # }
       

        elif self.language == "bn":
            self.process_funcs = {
                    'openbookqa': self.process_openbookqa_bn,
                    'arc-easy': self.process_arc_bn,
                    'arc-challenge': self.process_arc_bn,
                    'truthfulqa-mc': self.process_truthfulqa_mc_bn,
                    'truthfulqa-ml': self.process_truthfulqa_ml_bn,
                    'bbh-date': self.process_bbh_date,
                    'bbh-disamb-qa': self.process_bbh_general,
                    'bbh-geo-shapes':self.process_bbh_general,
                    'bbh-hyperbaton': self.process_bbh_general,
                    'bbh-logic-3': self.process_bbh_general,
                    'bbh-logic-5': self.process_bbh_general,
                    'bbh-logic-7': self.process_bbh_general,
                    'bbh-movie': self.process_bbh_general,
                    'bbh-penguins': self.process_bbh_general,
                    'bbh-reasoning': self.process_bbh_general,
                    'bbh-ruin-names': self.process_bbh_general,
                    'bbh-salient': self.process_bbh_general,
                    'bbh-snarks': self.process_bbh_general,
                    'bbh-temporal': self.process_bbh_general,
                    'bbh-track-3': self.process_bbh_general,
                    'bbh-track-5': self.process_bbh_general,
                    'bbh-track-7': self.process_bbh_general,
                    'commonsenseqa': self.process_commonsenseqa_bn,
                    'piqa': self.process_piqa_bn,
                    'mmlu': self.process_mmlu_bn,
                    'gsm8k-main': self.process_gsm8k,
                    'winogrande': self.process_winogrande_bn,
                    'boolq': self.process_boolq_bn,
                    'hellaswag':self.process_hellaswag_bn,
                    'titullm': self.process_titullm
                    # ... add more
                    }  
            
            # initilizing with default system_msg, overwrite if required
            self.sys_msg = {key: self.sys_msg_general_bn() for key in self.process_funcs.keys()}
            # overwrite system message for specific dataset
            self.sys_msg['gsm8k-main'] = self.sys_msg_gsm8k_bn()
            self.sys_msg['boolq'] = self.sys_msg_boolq_bn()
                            
            self.inp_msg = {
                            'openbookqa': self.input_msg_mcq_bn(4),
                            'arc-easy': self.input_msg_mcq_bn(5),
                            'arc-challenge': self.input_msg_mcq_bn(5),
                            'truthfulqa-mc': self.input_msg_mcq_bn(13),
                            'truthfulqa-ml': self.input_msg_truthfulqa_ml_bn,
                            'bbh-date': self.input_msg_mcq(4),
                            'bbh-disamb-qa': self.input_msg_mcq(4),
                            'bbh-geo-shapes':self.input_msg_mcq(4),
                            'bbh-hyperbaton': self.input_msg_mcq(4),
                            'bbh-logic-3': self.input_msg_mcq(4),
                            'bbh-logic-5': self.input_msg_mcq(4),
                            'bbh-logic-7': self.input_msg_mcq(4),
                            'bbh-movie': self.input_msg_mcq(4),
                            'bbh-penguins': self.input_msg_mcq(4),
                            'bbh-reasoning': self.input_msg_mcq(4),
                            'bbh-ruin-names': self.input_msg_mcq(4),
                            'bbh-salient': self.input_msg_mcq(4),
                            'bbh-snarks': self.input_msg_mcq(4),
                            'bbh-temporal': self.input_msg_mcq(4),
                            'bbh-track-3': self.input_msg_mcq(4),
                            'bbh-track-5': self.input_msg_mcq(4),
                            'bbh-track-7': self.input_msg_mcq(4),
                            'commonsenseqa': self.input_msg_mcq_bn(5),
                            'piqa': self.input_msg_mcq_bn(2),
                            'mmlu': self.input_msg_mcq_bn(4),
                            'gsm8k-main': self.input_msg_gsm8k_bn(),
                            'winogrande': self.input_msg_mcq_bn(2),
                            'boolq': self.input_msg_boolq_bn(),
                            'hellaswag':self.input_msg_mcq_bn(4),       
                            'titullm': self.input_msg_mcq(4)
                            # ... add more
                            }  
            # self.acc_func = {
            #     'openbookqa': self.acc_func_openbookqa  ,
            #     'arc-easy': self.acc_func_arc,
            #     'arc-challenge': self.acc_func_arc,
            #     'truthfulqa-mc': self.acc_func_truthfulqa_mc,
            #     'truthfulqa-ml': self.acc_func_truthfulqa_ml,
            #     'commonsenseqa': self.acc_func_commonsenseqa,
            # }


    ###############################
    ##### PROCESS FUNCTIONS #######
    ###############################
    

    # openbookqa 
    def process_openbookqa(self,input_text, question):
        question_text = question["question_stem"]
        choices_text = question["choices"]["text"]
        choices_label = question["choices"]["label"]

        input_text += f"{question_text}\nOptions:\n"

        options_list = []
        for label, choice in zip(choices_label, choices_text):
            input_text += f"{label}. {choice}\n"
            options_list.append(f"{label}. {choice}")

        label = question.get("answerKey", None)
        return input_text, label, question["id"]
    
    def process_openbookqa_bn(self,input_text, question):
        question_text = question["question_stem"]
        choices_text = ast.literal_eval(question["choices_text"])
        choices_label = ast.literal_eval(question["choices_label"])

        input_text += f"{question_text}\nবিকল্পসমূহ:\n"

        options_list = []
        for label, choice in zip(choices_label, choices_text):
            input_text += f"{label}. {choice}\n"
            options_list.append(f"{label}. {choice}")
        label = question.get("answerKey", None)
        return input_text, label, question["id"]
    
    # arc 
    def process_arc(self,input_text, question):
        question_text = question["question"]
        choices_text = question["choices"]["text"]
        choices_label = question["choices"]["label"]

        input_text += f"{question_text}\nOptions:\n"

        for label, choice in zip(choices_label, choices_text):
            input_text += f"{label}: {choice}\n"
        label = question.get("answerKey", None)
        return input_text,label,question['id']
    
    def process_arc_bn(self,input_text, question):
        question_text = question["question"]
        choices_dict = question["choices"]

        input_text += f"{question_text}\nবিকল্পসমূহ:\n"

        for choice_key, choice_value in choices_dict.items():
            input_text += f"{choice_key}: {choice_value}\n"

        label = question.get("answerKey", None)
        return input_text,label,question['id']

    # truthfulqa
    def process_truthfulqa_mc(self,input_text, question):
        question_text = question["question"]
        mc1_targets_choices = question["mc1_targets"]["choices"]  

        input_text += f"{question_text}\nOptions:\n"

        option_labels = [chr(65 + i) for i in range(len(mc1_targets_choices))]  

        for idx, choice in enumerate(mc1_targets_choices):
            option_text = f"{option_labels[idx]}: {choice}"
            input_text += f"{option_text}\n"
            
        mc1_targets_labels = question["mc1_targets"]["labels"]
        correct_idx = mc1_targets_labels.index(1) if 1 in mc1_targets_labels else None
        correct_option = chr(65 + correct_idx) if correct_idx is not None else None
            
        return input_text, correct_option, None
    
    def process_truthfulqa_ml(self,input_text, question):
        question_text = question["question"]
        mc2_targets_choices = question["mc2_targets"]["choices"]  

        input_text += f"{question_text}\nOptions:\n"

        option_labels = [chr(65 + i) for i in range(len(mc2_targets_choices))]  # A, B, C, ...

        for idx, choice in enumerate(mc2_targets_choices):
            option_text = f"{option_labels[idx]}: {choice}"
            input_text += f"{option_text}\n"
        
        mc2_targets_labels = question["mc2_targets"]["labels"]
        correct_indices = [i for i, label in enumerate(mc2_targets_labels) if label == 1]
        correct_options = [chr(65 + i) for i in correct_indices] if correct_indices else None
            
        return input_text, correct_options, None
    
    def process_truthfulqa_mc_bn(self,input_text, question):
        question_text = question["question"]
        mc1_targets_choices = question["mc1_targets"]["choices"]  

        input_text += f"{question_text}\nবিকল্পসমূহ:\n"

        option_labels = [chr(0x0995 + i) for i in range(len(mc1_targets_choices))]  # ক, খ, গ, ...

        for idx, choice in enumerate(mc1_targets_choices):
            option_text = f"{option_labels[idx]}: {choice}"
            input_text += f"{option_text}\n"
            
        mc1_targets_labels = question["mc1_targets"]["labels"]
        correct_idx = mc1_targets_labels.index(1) if 1 in mc1_targets_labels else None
        correct_option = chr(0x0995 + correct_idx) if correct_idx is not None else None
            
        return input_text,correct_option, None

    def process_truthfulqa_ml_bn(self,input_text, question):
        question_text = question["question"]
        mc2_targets_choices = question["mc2_targets"]["choices"]  

        input_text += f"{question_text}\nবিকল্পসমূহ:\n"

        option_labels = [chr(0x0995 + i) for i in range(len(mc2_targets_choices))]  # ক, খ, গ, ...

        for idx, choice in enumerate(mc2_targets_choices):
            option_text = f"{option_labels[idx]}: {choice}"
            input_text += f"{option_text}\n"
        
        mc2_targets_labels = question["mc2_targets"]["labels"]
        correct_indices = [i for i, label in enumerate(mc2_targets_labels) if label == 1]
        correct_options = [chr(0x0995 + i) for i in correct_indices] if correct_indices else None
            
        return input_text, correct_options, None
    
    
    # bbh
    def process_bbh_date(self,input_text, question):
        question_text = question["question"]
        options_list = question["options"]

        input_text += f"{question_text}\nOptions:\n"

        for option in options_list:
            input_text += f"{option}\n"

        return input_text


    def process_bbh_general(self,input_text, question):
        question_text = question["input"]
        options_list = question["options"]

        input_text += f"{question_text}\nOptions:\n"

        for option in options_list:
            input_text += f"{option}\n"

        return input_text

    # process_commonsenseqa
    def process_commonsenseqa(self,input_text, question):
        question_text = question["question"]
        question_concept = question.get("question_concept", "")
        choices_text = question["choices"]["text"]
        choices_label = question["choices"]["label"]

        input_text += f"{question_text}\n"

        if question_concept:
            input_text += f"Concept: {question_concept}\n"

        input_text += "Options:\n"

        for label, choice in zip(choices_label, choices_text):
            input_text += f"{label}. {choice}\n"

        label = question.get("answerKey", None)
        return input_text, label,question["id"]
    
    def process_commonsenseqa_bn(self,input_text, question):
        question_text = question["question"]
        question_concept = question.get("question_concept", "")
        choices_dict = question["choices"]
        
        input_text += f"{question_text}\n"
        
        if question_concept:
            input_text += f"প্রশ্নের বিষয়: {question_concept}\n"
        
        input_text += "বিকল্পসমূহ:\n"
        
        for choice_key, choice_value in choices_dict.items():
            input_text += f"{choice_key}: {choice_value}\n"
            
        label = question.get("answerKey", None)
        
        return input_text, label ,question["id"]
    
    # piqa
    def process_piqa(self,input_text, question):
        question_text = question["goal"]
        choices_dict = {"A": question["sol1"], "B": question["sol2"]}
        input_text += f"{question_text}\n"
        input_text += "Options:\n"
        for choice_key, choice_value in choices_dict.items():
            input_text += f"{choice_key}: {choice_value}\n"
            
        label = question.get("label", None)
        return input_text, label, None
    
    def process_piqa_bn(self,input_text, question):
        question_text = question["goal"]
        choices_dict = {"ক": question["sol1"], "খ": question["sol2"]}
        
        input_text += f"{question_text}\n"
        input_text += "বিকল্পসমূহ:\n"
        
        for choice_key, choice_value in choices_dict.items():
            input_text += f"{choice_key}: {choice_value}\n"
        label = question.get("label", None)
        return input_text, label, None

    # mmlu
    def process_mmlu(self,input_text, question):
        question_text = question["prompt"]
        choices_dict = {"A": question["A"], "B": question["B"], "C": question["C"], "D": question["D"]}
        
        input_text += f"{question_text}\n"
        input_text += "Options:\n"
        for choice_key, choice_value in choices_dict.items():
            input_text += f"{choice_key}: {choice_value}\n"
        label = question.get("answer", None)
        return input_text,label, None
    
    def process_mmlu_bn(self,input_text, question):
        question_text = question["prompt"]
        choices_dict = {"ক": question["A"], "খ": question["B"], "গ": question["C"], "ঘ": question["D"]}
        
        input_text += f"{question_text}\n"
        input_text += "বিকল্পসমূহ:\n"
        for choice_key, choice_value in choices_dict.items():
            input_text += f"{choice_key}: {choice_value}\n"
        label = question.get("answer", None)
        return input_text, label, None
    
    # gsm8k
    def process_gsm8k(self,input_text, question):
        question_text = question["question"]
        input_text += f"{question_text}\n"
        
        return input_text, question["answer"].split("#### ")[-1], None
    
    # winogrande
    def process_winogrande(self,input_text, question):
        question_text = question["sentence"]
        choices_dict = {"A": question["option1"], "B": question["option2"]}
        
        input_text += f"{question_text}\n"
        input_text += "Options:\n"
        
        for choice_key, choice_value in choices_dict.items():
            input_text += f"{choice_key}: {choice_value}\n"
        
        label = question.get("answer", None)
        return input_text, label,question["qID"]
    
    def process_winogrande_bn(self,input_text, question):
        question_text = question["sentence"]
        choices_dict = {"ক": question["option1"], "খ": question["option2"]}
        
        input_text += f"{question_text}\n"
        input_text += "বিকল্পসমূহ:\n"
        
        for choice_key, choice_value in choices_dict.items():
            input_text += f"{choice_key}: {choice_value}\n"
        label = question.get("answer", None)
        return input_text, label, question["qID"]
    
    # boolq
    def process_boolq(self,input_text, question):
        question_text = question["question"]
        passage_text = question["passage"]
        
        input_text += f"{question_text}\n"
        input_text += "Passage:\n"
        input_text += f"{passage_text}\n"
        
        label = question.get("answer", None)
        return input_text,label,None
    
    def process_boolq_bn(self,input_text, question):
        question_text = question["question"]
        passage_text = question["passage"]
        
        input_text += f"{question_text}\n"
        input_text += "অনুচ্ছেদ:\n"
        input_text += f"{passage_text}\n"
        
        label = question.get("answer", None)
        return input_text,label, None
    
    # hellaswag
    def process_hellaswag(self,input_text, question):
        english_characters = ["A", "B", "C", "D"]  
        input_text += "Given the context:\n"
        input_text += f"{question['ctx']}\n"
        input_text += "Which of the following is the most likely continuation?\n"
        
        endings_list = question["endings"]
        
        input_text += "Options:\n"
        
        for i, ending in enumerate(endings_list):
            input_text += f"{english_characters[i]}. {ending}\n"
            
        label = question.get("label", None)
        
        return input_text, label, None
    
    
    def process_hellaswag_bn(self,input_text, question):
        bangla_characters = ["ক", "খ", "গ", "ঘ", "ঙ"]  

        bangla_replacements = {
            "Header": "হেডার",
            "Title": "টাইটেল", 
            "Step":"স্টেপ" 
        }
        for key, value in bangla_replacements.items():
            question['ctx'] = question['ctx'].replace(key, value)
        
        input_text += "প্রদত্ত প্রসঙ্গ:\n"
        input_text += f"{question['ctx']}\n"
        input_text += "নিচের কোনটি সবচেয়ে সম্ভাব্য পরবর্তী অংশ?\n"
        
        endings_list = question["endings"]
        
        input_text += "বিকল্পসমূহ:\n"
        
        for i, ending in enumerate(endings_list):
            input_text += f"{bangla_characters[i]}. {ending}\n" 
            
        label = question.get("label", None)
        
        return input_text, label, None
    
    
    
    # titullm
    def process_titullm(self,input_text, question):
        question_text = question["question_stem"]
        choices_text = ast.literal_eval(question["choices_text"])
        choices_label = ast.literal_eval(question["choices_label"])

        input_text += f"{question_text}\nOptions:\n"

        options_list = []
        for label, choice in zip(choices_label, choices_text):
            input_text += f"{label}. {choice}\n"
            options_list.append(f"{label}. {choice}")

        return input_text, options_list
 
    # add more process funcs
    
    ###############################
    ##### SYSTEM MESSAGES #########
    ###############################
    
    # general
    def sys_msg_general(self):
        SYSTEM_MESSAGE = (
        "You are a highly intelligent English-language assistant designed to analyze questions and "
        "provide concise answers by selecting the most accurate label from a given list."
    )
        return SYSTEM_MESSAGE
    
    def sys_msg_general_bn(self):
        SYSTEM_MESSAGE = (
        "আপনি একজন অত্যন্ত বুদ্ধিমান  বাংলা ভাষার সহকারী, যিনি প্রশ্ন বিশ্লেষণ করে একটি প্রদত্ত তালিকা থেকে সবচেয়ে সঠিক লেবেল নির্বাচন করে সংক্ষিপ্ত উত্তর প্রদান করেন।"      
   )
        return SYSTEM_MESSAGE
    
    # openbookqa
    def sys_msg_openbookqa(self):
        return self.sys_msg_general()
    # arc
    def sys_msg_arc(self):
        return self.sys_msg_general()
    
    # gsm8k
    def sys_msg_gsm8k(self):
        SYSTEM_MESSAGE = (
        "You are a highly intelligent English-language assistant designed to solve mathematical questions "
        "and provide only english numeric answers based on accurate calculations without any explanation. "
        ) 
        return SYSTEM_MESSAGE
    def sys_msg_gsm8k_bn(self):
        SYSTEM_MESSAGE = (
        "আপনি একজন অত্যন্ত বুদ্ধিমান বাংলা-ভাষার সহকারী, যার কাজ হলো গণিত সম্পর্কিত প্রশ্ন সমাধান করা "
        "এবং নির্ভুল গণনার ভিত্তিতে শুধুমাত্র বাংলা সংখ্যায় উত্তর প্রদান করা, কোনো ব্যাখ্যা ছাড়া।"
        )
        return SYSTEM_MESSAGE
    
    # boolq
    def sys_msg_boolq(self):
        SYSTEM_MESSAGE = (
        "You are a highly intelligent assistant designed to analyze yes/no questions and "
        "provide concise answers by selecting the most accurate label (true or false) based on the given passage."
        )
        return SYSTEM_MESSAGE
    
    def sys_msg_boolq_bn(self):
        SYSTEM_MESSAGE = (
        "আপনি একটি অত্যন্ত বুদ্ধিমান সহকারী, যা সত্য/মিথ্যা প্রশ্ন বিশ্লেষণ করতে সক্ষম এবং "
        "প্রদত্ত অনুচ্ছেদের ভিত্তিতে সবচেয়ে নির্ভরযোগ্য লেবেল (সত্য বা মিথ্যা) নির্বাচন করে সংক্ষিপ্ত উত্তর প্রদান করে।"
        )
        return SYSTEM_MESSAGE
    
    # add more customized sys msg function it could be datasetspecific or general

    ################################
    ##### INPUT MESSAGES ###########
    ################################
    def input_msg_mcq(self, num_option = 4):
        if num_option == 2:
            INPUT_MESSAGE = (
            "Please select the most appropriate answer. "
            "Respond with the label (e.g., A, B) without any explanation or additional text.\n"
            "Question: "
            )
        
        if num_option == 4:
            INPUT_MESSAGE = (
                "Please select the most appropriate answer. "
                "Respond with the label (e.g., A, B, C, D) without any explanation or additional text.\n"
                "Question: "
            )
        if num_option == 5:
            INPUT_MESSAGE = (
            "Please select the most appropriate answer. "
            "Respond with the label (e.g., A, B, C, D, E) without any explanation or additional text.\n"
            "Question: "
            )
            
        if num_option == 13:
            INPUT_MESSAGE = (
            "Please select the most appropriate answer. "
            "Respond with the label (e.g., A, B, C, D, E,F,G,H,I,J,K,L,M) without any explanation or additional text.\n"
            "Question: "
            )
        return INPUT_MESSAGE
    
    def input_msg_mcq_bn(self, num_option = 4):
        if num_option == 2:
            INPUT_MESSAGE = (
            "অনুগ্রহ করে সবচেয়ে উপযুক্ত উত্তরটি নির্বাচন করুন। "
            "কোনো ব্যাখ্যা বা অতিরিক্ত লেখা ছাড়া শুধুমাত্র লেবেল (যেমন, ক, খ) দিয়ে উত্তর দিন।\n"
            "প্রশ্ন: "
            )
        if num_option == 4:
            INPUT_MESSAGE = (
            "অনুগ্রহ করে সবচেয়ে উপযুক্ত উত্তরটি নির্বাচন করুন। "
            "কোনো ব্যাখ্যা বা অতিরিক্ত লেখা ছাড়া শুধুমাত্র লেবেল (যেমন, ক, খ, গ, ঘ) দিয়ে উত্তর দিন।\n"
            "প্রশ্ন: "
            )
        if num_option == 5:
            INPUT_MESSAGE = (
             "অনুগ্রহ করে সবচেয়ে উপযুক্ত উত্তরটি নির্বাচন করুন। "
            "কোনো ব্যাখ্যা বা অতিরিক্ত লেখা ছাড়া শুধুমাত্র লেবেল (যেমন, ক, খ, গ, ঘ, ঙ) দিয়ে উত্তর দিন।\n"
            "প্রশ্ন: "
            )
            
        if num_option == 13:
            INPUT_MESSAGE = (
             "অনুগ্রহ করে সবচেয়ে উপযুক্ত উত্তরটি নির্বাচন করুন। "
            "কোনো ব্যাখ্যা বা অতিরিক্ত লেখা ছাড়া শুধুমাত্র লেবেল (যেমন, ক, খ, গ, ঘ, ঙ, চ, ছ, জ, ঝ, ঞ, ট, ঠ, ড) দিয়ে উত্তর দিন।\n"
            "প্রশ্ন: "
            )
        return INPUT_MESSAGE
    
    
    # openbookqa
    def inp_msg_openbookqa(self):
        return self.input_msg_mcq()
        # arc
    def inp_msg_arc(self):
        return self.input_msg_mcq()
    # add more customized input msg function it could be datasetspecific or general
    def input_msg_truthfulqa_ml(self):
        INPUT_MESSAGE_MULTI = (
        "Please select all the appropriate answers. "
        "Respond with all the correct labels as a list separated by comma(s) (e.g., [A,B,C,D,E,F,G,H,I,J,K,L,M,N]) without any explanation or additional text. \n"
        "Question: "
        )
        return INPUT_MESSAGE_MULTI
    
    def input_msg_truthfulqa_ml_bn(self):
        INPUT_MESSAGE_MULTI = (
        "অনুগ্রহ করে সমস্ত উপযুক্ত উত্তর নির্বাচন করুন। "
        "সঠিক লেবেলগুলিকে কমা দ্বারা পৃথক করে একটি তালিকা হিসাবে উত্তর দিন (যেমন, [ক, খ, গ, ঘ, ঙ, চ, ছ, জ, ঝ, ঞ, ট, ঠ, ড, ঢ]) কোনো ব্যাখ্যা বা অতিরিক্ত লেখা ছাড়া।\n"
        "প্রশ্ন: "
        )   
        return INPUT_MESSAGE_MULTI
    
    def input_msg_gsm8k(self):
        INPUT_MESSAGE = (
        "Please provide the most appropriate English numeric answer based on the question."
        "If the answer requires multi-step reasoning, execute them concisely, and encompass it within a <reason>  tag." 
        "Irrespective of reasoning steps, you must explicitly state the answer at the end within an <answer> tag"
        "For example, if the answer is 300, respond with only: <answer>300</answer>\n"
        "Question: "
        )
        return INPUT_MESSAGE
    
    def input_msg_gsm8k_bn(self):
        INPUT_MESSAGE = (
        "দয়া করে প্রশ্নের ভিত্তিতে সবচেয়ে উপযুক্ত বাংলা সংখ্যায় উত্তর প্রদান করুন। "
        "যদি উত্তর বের করতে একাধিক ধাপের গণনা প্রয়োজন হয়, তাহলে সংক্ষেপে সেই ধাপগুলো <কারণ> ট্যাগের মধ্যে রাখুন। "
        "গণনার ধাপ যাই হোক, আপনাকে অবশ্যই চূড়ান্ত উত্তর <উত্তর> ট্যাগের মধ্যে প্রকাশ করতে হবে। "
        "উদাহরণস্বরূপ, যদি বাংলা উত্তর হয় ৩০০, তাহলে শুধুমাত্র এইভাবে উত্তর দিন: <উত্তর>৩০০</উত্তর>\n"
        "প্রশ্ন: "
        )
        return INPUT_MESSAGE
    
    def input_msg_boolq(self):
        INPUT_MESSAGE = (
        "Please select the most appropriate answer. "
        "Respond with the label (e.g.,true/false) without any explanation or additional text.\n"
        "Question: "
        )
        return INPUT_MESSAGE  
    
    def input_msg_boolq_bn(self):
        INPUT_MESSAGE = (
        "সবচেয়ে উপযুক্ত উত্তর নির্বাচন করুন। "
        "লেবেল (যেমন, সত্য/মিথ্যা) দিয়ে উত্তর দিন, কোনও ব্যাখ্যা বা অতিরিক্ত পাঠ্য যোগ করবেন না।\n"
        "প্রশ্ন: "
        )
        return INPUT_MESSAGE  
 
    
    #############################
    ###### get funcs ############
    #############################
    def get_process_func(self, dataset_name):
        return self.process_funcs[dataset_name]
    
    def get_sys_msg(self, dataset_name):
        return self.sys_msg[dataset_name]
    
    def get_inp_msg(self, dataset_name):
        return self.inp_msg[dataset_name]
    # def get_acc_func(self, dataset_name):
    #     return self.acc_func[dataset_name]
    # def get_rer_func(self, dataset_name):
    #     return self.rer_func[dataset_name]

def clean_response(response):
    if "<think>" in response and "</think>" in response:
        response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)  # Remove <think> content
    return response.strip()

def parse_response_rer(input_csv):
    options_lists = []
    csv.field_size_limit(1000000) 
    with open(input_csv, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)   
        for row in reader:
            options = [
                opt.strip()
                for opt in row["Prompt"].split("\n")
                if opt.startswith(("A", "B", "C", "D", "E"," F", "G", "H", "I", "J", "K", "L", "M", "N")) or opt.startswith(("ক", "খ", "গ", "ঘ", "ঙ","চ", "ছ", "জ", "ঝ", "ঞ", "ট", "ঠ", "ড", "ঢ")) or opt.startswith(("true","false")) or opt.startswith(("সত্য","মিথ্যা"))
            ]      
            options_lists.append(options)
    return options_lists

def parse_response(input_csv, column):
    responses = []
    csv.field_size_limit(1000000)
    with open(input_csv, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            cleaned_response = clean_response(row[column])
            responses.append(cleaned_response)
    return responses

 
