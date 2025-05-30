import re

label_mapping_boolq = {
    "সত্য": "true",
    "মিথ্যা": "false",
}

label_mapping_cqsa = {"ক": "A", "খ": "B", "গ": "C", "ঘ": "D", "ঙ": "E"}
label_mapping_winogrande_bn = {
    "ক": "1",
    "খ": "2",
}

label_mapping_winogrande_en = {
    "A": "1",
    "B": "2",
}

label_mapping_hellaswag_bn = {
    "ক": "0",
    "খ": "1",
    "গ": "2",
    "ঘ": "3"
}
label_mapping_hellaswag_en = {
    "A": "0",
    "B": "1",
    "C": "2",
    "D": "3"
}


def extract_response_gsm8k(text, lang):
    if lang == "bn":
        match = re.search(r"<উত্তর>(.*?)</উত্তর>", text)
        return match.group(1) if match else None
    else:
        match = re.search(r"<answer>(.*?)</answer>", text)
        return match.group(1) if match else None


def accuracy(response, answer, dataset=None, lang=None):
    if dataset == "boolq":
        response = [label_mapping_boolq.get(txt, txt) for txt in response]
        response = [txt.lower() for txt in response]
        answer = [txt.lower() for txt in answer]
        results = [txt == ans for txt, ans in zip(response, answer)]
        metric_val = sum(results) / len(results)
        return metric_val

    elif dataset == "gsm8k":
        extracted_response = [extract_response_gsm8k(resp, lang) for resp in response]
        results = [resp == ans for resp, ans in zip(extracted_response, answer)]
        metric_val = sum(results) / len(results)
        return metric_val

    elif dataset == "cqsa" and lang == "bn":
        mapped_response = [label_mapping_cqsa.get(txt, txt) for txt in response]
        results = [txt == ans for txt, ans in zip(mapped_response, answer)]

        metric_val = sum(results) / len(results)
        return metric_val

    elif dataset == "winogrande" and lang == "bn":
        response = [label_mapping_winogrande_bn.get(txt, txt) for txt in response]
        results = [txt == ans for txt, ans in zip(response, answer)]
        metric_val = sum(results) / len(results)
        return metric_val
    
    elif dataset == "winogrande" and lang == "en":
        response = [label_mapping_winogrande_en.get(txt, txt) for txt in response]
        results = [txt == ans for txt, ans in zip(response, answer)]
        metric_val = sum(results) / len(results)
        return metric_val
    
    elif dataset == "hellaswag" and lang == "en":
        response = [label_mapping_hellaswag_en.get(txt, txt) for txt in response]
        response = [txt.lower() for txt in response]
        answer = [txt.lower() for txt in answer]
        results = [txt == ans for txt, ans in zip(response, answer)]
        metric_val = sum(results) / len(results)
        return metric_val
    
    elif dataset == "hellaswag" and lang == "bn":
        response = [label_mapping_hellaswag_bn.get(txt, txt) for txt in response]
        response = [txt.lower() for txt in response]
        answer = [txt.lower() for txt in answer]
        results = [txt == ans for txt, ans in zip(response, answer)]
        metric_val = sum(results) / len(results)
        return metric_val

    else:
        results = [txt == ans for txt, ans in zip(response, answer)]
        metric_val = sum(results) / len(results)
        return metric_val


def response_error_rate(response, options, dataset=None, lang=None):
    if dataset == "boolq":
        response = [label_mapping_boolq.get(txt, txt) for txt in response]
        response = [txt.lower() for txt in response]
        results = []
        for txt, opt_list in zip(response, options):
            cond = not any(opt == txt for opt in opt_list)
            results.append(cond)
        metric_val = sum(results) / len(results)
        return metric_val

    elif dataset == "gsm8k":
        number_pattern = None
        if lang == "en":
            number_pattern = re.compile(r"^[০-৯]+$")
        else:
            number_pattern = re.compile(r"^[0-9]+$")
        results = []
        extracted_response = [extract_response_gsm8k(resp, lang) for resp in response]
        for txt in extracted_response:
            if txt is None or not isinstance(txt, str):
                results.append(True)  # Treat null or non-string as an error
                continue
            cleaned_txt = txt.replace(",", "").replace(" ", "")
            cond = not number_pattern.match(cleaned_txt)
            results.append(cond)
        metric_val = sum(results) / len(results)
        return metric_val
    
    elif dataset == "cqsa" and lang == "bn":
        mapped_response = [label_mapping_cqsa.get(txt, txt) for txt in response]
        results = []
        for txt, opt_list in zip(mapped_response, options):
            cond = not any(label_mapping_cqsa.get(opt[0]) == txt for opt in opt_list)
            results.append(cond)
        metric_val = sum(results) / len(results)
        return metric_val

    elif dataset == "winogrande" and lang == "bn":
        results = []
        opt_list = ["ক", "খ"]
        for txt in response:
            cond = not any(opt == txt for opt in opt_list)
            results.append(cond)
        metric_val = sum(results) / len(results)
        return metric_val
    
    elif dataset == "winogrande" and lang == "en":
        results = []
        opt_list = ["A", "B"]
        for txt in response:
            cond = not any(opt == txt for opt in opt_list)
            results.append(cond)
        metric_val = sum(results) / len(results)
        return metric_val
    
    elif dataset == "hellaswag" and lang == "bn":
        response = [txt.lower() for txt in response]
        results = []
        opt_list = ["ক", "খ", "গ", "ঘ"]
        for txt in response:
            cond = not any(opt == txt for opt in opt_list)
            results.append(cond)
        metric_val = sum(results) / len(results)
        return metric_val
    
    elif dataset == "hellaswag" and lang == "en":
        response = [txt.lower() for txt in response]
        results = []
        opt_list = ["A", "B", "C", "D"]
        for txt in response:
            cond = not any(opt.lower() == txt for opt in opt_list)
            results.append(cond)
        metric_val = sum(results) / len(results)
        return metric_val

    else:
        results = []
        for txt, opt_list in zip(response, options):
            cond = not any(opt.startswith(txt) for opt in opt_list)
            results.append(cond)
        metric_val = sum(results) / len(results)
        return metric_val
