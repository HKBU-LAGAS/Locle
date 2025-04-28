import tiktoken
import openai
import json
import os.path as osp
import asyncio
from openail.asyncapi import process_api_requests_from_file
import torch
import logging
import ast
import matplotlib.pyplot as plt
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 
import numpy as np
import re





async def call_async_api(request_filepath, save_filepath, request_url, api_key, max_request_per_minute, max_tokens_per_minute, sp, ss):
    await process_api_requests_from_file(
            requests_filepath=request_filepath,
            save_filepath=save_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_request_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name='cl100k_base',
            max_attempts=int(2),
            logging_level=int(logging.INFO),
            seconds_to_pause=sp,
            seconds_to_sleep=ss
        )


def set_endpoints(openai_key, openai_endpoints):
    openai.api_key = openai_key
    openai.api_base = openai_endpoints



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_text_api_with_backoff(*args, **kwargs):
    return openai_text_api(*args, **kwargs)


def openai_text_api(input_text, api_key, model_name = "gpt-3.5-turbo", temperature = 0, n = 1):
    openai.api_base = "change to your endpoint"
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": input_text}],
        temperature=temperature,
        api_key=api_key,
        n = n)
    return response 

def openai_text_api_with_top_p(input_text, api_key, model_name = "gpt-3.5-turbo", top_p = 1, n = 1):
    openai.api_base = "change to your endpoint"
    response = openai.ChatCompletion.create(
        model = model_name, 
        messages=[{"role": "user", "content": input_text}],
        top_p = top_p,
        api_key = api_key,
        n = n
    )
    return response


def generate_chat_input_file(input_text, model_name = 'gpt-3.5-turbo', temperature = 0, n = 1):
    jobs = []
    for i, text in enumerate(input_text):
        obj = {}
        obj['model'] = model_name
        obj['messages'] = [
            {
                'role': 'user',
                'content': text 
            }
        ]
        obj['temperature'] = temperature
        obj['n'] = n
        jobs.append(obj)
    return jobs 

def my_efficient_openai_text_api(input_text, filename, savepath, sp, ss, api_key="change to your key", request_url = "change to your endpoint", rewrite = True, temperature = 0.0, n = 1, label_num = 7, label_name=None, node2_labels=None, use_type='one_dim', cur_node=0):
    # compare with each sample label and get the most similar one 
    rewrite = True
    import re
    non_empty_results = []
    pred_results = -1
    from llm import calculate_cost_from_a_list_of_texts
    token_cost = calculate_cost_from_a_list_of_texts(input_text)
    print(token_cost)
    while token_cost > 4000:
        import re
        paper_content = re.search(r"(Target paper:)(.*?)(Output)", input_text[0], re.DOTALL)
        paper_content = paper_content.group(2).strip()
        split_paper_content = paper_content[:int(len(paper_content)*0.9)]
        input_text[0] = input_text[0].replace(paper_content, split_paper_content)
        token_cost = calculate_cost_from_a_list_of_texts(input_text)
    if not osp.exists(savepath) or rewrite:
        jobs = generate_chat_input_file(input_text, model_name='gpt-3.5-turbo', temperature = temperature, n = n)
        with open(filename, "w") as f:
            for i, job in enumerate(jobs):
                json_string = json.dumps(job)
                if job['messages'][0]['content'] != "":
                    f.write(json_string + "\n")
                    non_empty_results.append(i)
        asyncio.run(
            call_async_api(
                filename, save_filepath=savepath,
                request_url=request_url,
                api_key=api_key,
                max_request_per_minute=200, 
                max_tokens_per_minute=9000000000,
                sp=sp,
                ss=ss
            )
        )
    # import pdb; pdb.set_trace()
    openai_result = []
    with open(savepath, 'r') as f:
        pred_label = -1
        pred_confidence = -1
        labels = [0] * label_num
        for line in f:
            json_obj = json.loads(line.strip())
            content = json_obj[1]
            idx = json_obj[-1]
            choices = []
            Yes_num = 0
            No_num = 0
            if content == "":
                pass
            elif isinstance(idx, int):
                try:
                    if use_type == 'two_dim':
                        choices = [x['message']['content'] for x in json_obj[1]['choices']]
                        for choice in choices:
                            answer = choice.split(",")[0]
                            if answer == "Yes":
                                Yes_num += 1
                            elif answer == "No":
                                No_num += 1
                        if Yes_num > No_num:
                            labels[node2_labels[idx]] += 1
                    elif use_type == 'one_dim':
                        choices = [x['message']['content'] for x in json_obj[1]['choices']]
                        for choice in choices:
                            choice = re.sub(r'[\[\]\{\}\"]', '', choice)
                            answer = choice.split(": ")[1].split(",")[0]
                            pred_confidence = float(choice.split(": ")[-1].replace(" ", ''))
                            for i, label in enumerate(label_name):
                                if answer.lower() == label.lower():
                                    labels[i] += 1
                except:
                    # import ipdb; ipdb.set_trace()
                    continue
            else:
                import ipdb; ipdb.set_trace()
                idx = json_obj[-2]
                new_result = openai_text_api(json_obj[0]['messages'][0]['content'], api_key, model_name = json_obj[0]['model'], temperature = json_obj[0]['temperature'], n = json_obj[0]['n'])
                try:
                    choices = [json.loads(x['message']['content']) for x in new_result[1]['choices']]
                    for choice in choices:
                        for select in choice:
                            match = re.search(r'\d+', select['answer'])
                            if match:
                                number = int(match.group())
                                labels[select['confidence']] += number
                except:
                    # import ipdb; ipdb.set_trace()
                    continue
            
    # To do: if found no answer, use llm to directly predict the answer
    # import pdb; pdb.set_trace()
    if use_type == 'two_dim':
        for i in range(label_num):
            label_num = torch.where(node2_labels==i)[0].shape[0]
            if label_num > 0:
                labels[i] = labels[i] / label_num
        print(f"cluster node:{np.argmax(labels)}, confidence:{labels[np.argmax(labels)]}")
        return np.argmax(labels), labels[np.argmax(labels)]
   
    elif use_type == 'one_dim':
        pred_label = np.argmax(labels)
        if labels[pred_label] == 0:
            # import pdb; pdb.set_trace()
            return -1, -1
        else:
            return pred_label, pred_confidence

def my_efficient_openai_text_api_label(input_text, filename, savepath, sp, ss, api_key="change to your key", request_url = "change to your endpoint", rewrite = True, temperature = 0.0, n = 1, label_num = 7, label_name=None, node2_labels=None):
    # compare with each sample label and get the most similar one 
    rewrite = True
    non_empty_results = []
    pred_results = -1

    if not osp.exists(savepath) or rewrite:
        jobs = generate_chat_input_file(input_text, model_name='gpt-3.5-turbo', temperature = temperature, n = n)
        with open(filename, "w") as f:
            for i, job in enumerate(jobs):
                json_string = json.dumps(job)
                if job['messages'][0]['content'] != "":
                    f.write(json_string + "\n")
                    non_empty_results.append(i)
        asyncio.run(
            call_async_api(
                filename, save_filepath=savepath,
                request_url=request_url,
                api_key=api_key,
                max_request_per_minute=100000, 
                max_tokens_per_minute=9000000000,
                sp=sp,
                ss=ss
            )
        )
    # import pdb; pdb.set_trace()
    openai_result = []
    with open(savepath, 'r') as f:
        pred_label = -1
        pred_confidence = -1

        for line in f:
            labels = [0] * label_num
            json_obj = json.loads(line.strip())
            content = json_obj[1]
            idx = json_obj[-1]
            choices = []
            Yes_num = 0
            No_num = 0
            if content == "":
                pass
            elif isinstance(idx, int):
                try:
                    choices = [x['message']['content'] for x in json_obj[1]['choices']]
                    for choice in choices:
                        answer = choice.split(": ")[1]
                        for idx, name in enumerate(label_name):
                            if name == answer:
                                labels[idx] += 1
                except:
                    # import ipdb; ipdb.set_trace()
                    continue
            else:
                import ipdb; ipdb.set_trace()
                idx = json_obj[-2]
                new_result = openai_text_api(json_obj[0]['messages'][0]['content'], api_key, model_name = json_obj[0]['model'], temperature = json_obj[0]['temperature'], n = json_obj[0]['n'])
                try:
                    choices = [json.loads(x['message']['content']) for x in new_result[1]['choices']]
                    for choice in choices:
                        for select in choice:
                            match = re.search(r'\d+', select['answer'])
                            if match:
                                number = int(match.group())
                                labels[select['confidence']] += number
                except:
                    # import ipdb; ipdb.set_trace()
                    continue
            
    # To do: if found no answer, use llm to directly predict the answer
    pred_label = np.argmax(labels)
    pred_value = labels[np.argmax(labels)]
    if pred_value > 0:
        return pred_label
    else:
        return -1

def efficient_openai_text_api(input_text, filename, savepath, sp, ss, api_key="change_this_to_your_key", request_url = "change to your endpoint", rewrite = True, temperature = 0, n = 1):
    # openai_result = []
    rewrite = True
    non_empty_results = []
    if not osp.exists(savepath) or rewrite:
        jobs = generate_chat_input_file(input_text, model_name='gpt-3.5-turbo', temperature = temperature, n = n)
        with open(filename, "w") as f:
            for i, job in enumerate(jobs):
                json_string = json.dumps(job)
                if job['messages'][0]['content'] != "":
                    # import pdb; pdb.set_trace()
                    f.write(json_string + "\n")
                    non_empty_results.append(i)
        asyncio.run(
            call_async_api(
                filename, save_filepath=savepath,
                request_url=request_url,
                api_key=api_key,
                max_request_per_minute=100000, 
                max_tokens_per_minute=100000,
                sp=sp,
                ss=ss,
            )
        )
    openai_result = []
    with open(savepath, 'r') as f:
        # import ipdb; ipdb.set_trace()
        for line in f:
            json_obj = json.loads(line.strip())
            content = json_obj[1]
            idx = json_obj[-1]
            choices = []
            if content == "":
                openai_result.append(("", idx))
                # import ipdb; ipdb.set_trace()
            elif isinstance(idx, int):
                choices = [x['message']['content'] for x in json_obj[1]['choices']]
                openai_result.append((choices, idx))
            else:
                idx = json_obj[-2]
                new_result = openai_text_api(json_obj[0]['messages'][0]['content'], api_key, model_name = json_obj[0]['model'], temperature = json_obj[0]['temperature'], n = json_obj[0]['n'])
                choices = [x['message']['content'] for x in new_result['choices']]
                openai_result.append((choices, idx))
    openai_result = sorted(openai_result, key=lambda x:x[-1])
    results = [("", idx) for idx in range(len(input_text))]
    for i, r in enumerate(openai_result):
        results[non_empty_results[i]] = r
    return results
    

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-16k":
        print("Warning: gpt-3.5-turbo-16k may change over time. Returning num tokens assuming gpt-3.5-turbo-16k-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-16k-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_tokens_from_string(string: str, model = "text-davinci-003") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def load_partial_openai_result(data_path, dataset_name, prompt_key):
    # import pdb; pdb.set_trace()
    file_name = osp.join(data_path, f"{dataset_name}_openai.pt")
    if osp.exists(file_name):
        return torch.load(file_name, map_location='cpu')
    else:
        return None



def save_partial_openai_result(data_path, dataset_name, data, key = 'zero_shot', load_pre_existing = None, num_of_elements = -1, rewrite = True):
    if not osp.exists(osp.join(data_path, f"{dataset_name}_openai.pt")):
        ## create new datastore
        datastore = {}
        datastore['zero_shot'] = ["" for _ in range(num_of_elements)]
        if load_pre_existing != None:
            pre_existing = torch.load(osp.join(data_path, pre_existing))
            datastore['zero_shot'] = pre_existing
    else:
        datastore = load_partial_openai_result(data_path, dataset_name, key)
        if datastore == None:
            datastore = {}
            datastore['zero_shot'] = ["" for _ in range(num_of_elements)]
            if load_pre_existing != None:
                pre_existing = torch.load(osp.join(data_path, pre_existing))
                datastore['zero_shot'] = pre_existing

    if datastore.get(key) == None:
        datastore[key] = ["" for _ in range(num_of_elements)]

    for i in range(len(data)):
        if (data[i] != "" and datastore[key][i] == "") or (rewrite and data[i] != ""):
            datastore[key][i] = data[i]
    torch.save(datastore, osp.join(data_path, f"{dataset_name}_openai.pt"))

        

def load_mapping_2():
    # arxiv_mapping = {
    # 'arxiv cs ai': 'cs.AI',
    # 'arxiv cs cl': 'cs.CL',
    # 'arxiv cs cc': 'cs.CC',
    # 'arxiv cs ce': 'cs.CE',
    # 'arxiv cs cg': 'cs.CG',
    # 'arxiv cs gt': 'cs.GT',
    # 'arxiv cs cv': 'cs.CV',
    # 'arxiv cs cy': 'cs.CY',
    # 'arxiv cs cr': 'cs.CR',
    # 'arxiv cs ds': 'cs.DS',
    # 'arxiv cs db': 'cs.DB',
    # 'arxiv cs dl': 'cs.DL',
    # 'arxiv cs dm': 'cs.DM',
    # 'arxiv cs dc': 'cs.DC',
    # 'arxiv cs et': 'cs.ET',
    # 'arxiv cs fl': 'cs.FL',
    # 'arxiv cs gl': 'cs.GL',
    # 'arxiv cs gr': 'cs.GR',
    # 'arxiv cs ar': 'cs.AR',
    # 'arxiv cs hc': 'cs.HC',
    # 'arxiv cs ir': 'cs.IR',
    # 'arxiv cs it': 'cs.IT',
    # 'arxiv cs lo': 'cs.LO',
    # 'arxiv cs lg': 'cs.LG',
    # 'arxiv cs ms': 'cs.MS',
    # 'arxiv cs ma': 'cs.MA',
    # 'arxiv cs mm': 'cs.MM',
    # 'arxiv cs ni': 'cs.NI',
    # 'arxiv cs ne': 'cs.NE',
    # 'arxiv cs na': 'cs.NA',
    # 'arxiv cs os': 'cs.OS',
    # 'arxiv cs oh': 'cs.OH',
    # 'arxiv cs pf': 'cs.PF',
    # 'arxiv cs pl': 'cs.PL',
    # 'arxiv cs ro': 'cs.RO',
    # 'arxiv cs si': 'cs.SI',
    # 'arxiv cs se': 'cs.SE',
    # 'arxiv cs sd': 'cs.SD',
    # 'arxiv cs sc': 'cs.SC',
    # 'arxiv cs sy': 'cs.SY'
    # }

    arxiv_mapping = {'arxiv cs ai': 'Artificial Intelligence', 'arxiv cs cl': 'Computation and Language', 'arxiv cs cc': 'Computational Complexity', 'arxiv cs ce': 'Computational Engineering, Finance, and Science', 'arxiv cs cg': 'Computational Geometry', 'arxiv cs gt': 'Computer Science and Game Theory', 'arxiv cs cv': 'Computer Vision and Pattern Recognition', 'arxiv cs cy': 'Computers and Society', 'arxiv cs cr': 'Cryptography and Security', 'arxiv cs ds': 'Data Structures and Algorithms', 'arxiv cs db': 'Databases', 'arxiv cs dl': 'Digital Libraries', 'arxiv cs dm': 'Discrete Mathematics', 'arxiv cs dc': 'Distributed, Parallel, and Cluster Computing', 'arxiv cs et': 'Emerging Technologies', 'arxiv cs fl': 'Formal Languages and Automata Theory', 'arxiv cs gl': 'General Literature', 'arxiv cs gr': 'Graphics', 'arxiv cs ar': 'Hardware Architecture', 'arxiv cs hc': 'Human-Computer Interaction', 'arxiv cs ir': 'Information Retrieval', 'arxiv cs it': 'Information Theory', 'arxiv cs lo': 'Logic in Computer Science', 'arxiv cs lg': 'Machine Learning', 'arxiv cs ms': 'Mathematical Software', 'arxiv cs ma': 'Multiagent Systems', 'arxiv cs mm': 'Multimedia', 'arxiv cs ni': 'Networking and Internet Architecture', 'arxiv cs ne': 'Neural and Evolutionary Computing', 'arxiv cs na': 'Numerical Analysis', 'arxiv cs os': 'Operating Systems', 'arxiv cs oh': 'Other Computer Science', 'arxiv cs pf': 'Performance', 'arxiv cs pl': 'Programming Languages', 'arxiv cs ro': 'Robotics', 'arxiv cs si': 'Social and Information Networks', 'arxiv cs se': 'Software Engineering', 'arxiv cs sd': 'Sound', 'arxiv cs sc': 'Symbolic Computation', 'arxiv cs sy': 'Systems and Control'}
    citeseer_mapping = {
        "Agents": "Agents",
        "ML": "Machine Learning",
        "IR": "Information Retrieval",
        "DB": "Database",
        "HCI": "Human Computer Interaction",
        "AI": "Artificial Intelligence"
    }
    pubmed_mapping = {
        'Diabetes Mellitus, Experimental': 'Diabetes Mellitus Experimental',
        'Diabetes Mellitus Type 1': 'Diabetes Mellitus Type 1',
        'Diabetes Mellitus Type 2': 'Diabetes Mellitus Type 2'
    }
    cora_mapping = {
        'Rule_Learning': "Rule_Learning",
        'Neural_Networks': "Neural_Networks",
        'Case_Based': "Case_Based",
        'Genetic_Algorithms': "Genetic_Algorithms",
        'Theory': "Theory",
        'Reinforcement_Learning': "Reinforcement_Learning",
        'Probabilistic_Methods': "Probabilistic_Methods"
    }

    products_mapping = {'Home & Kitchen': 'Home & Kitchen',
        'Health & Personal Care': 'Health & Personal Care',
        'Beauty': 'Beauty',
        'Sports & Outdoors': 'Sports & Outdoors',
        'Books': 'Books',
        'Patio, Lawn & Garden': 'Patio, Lawn & Garden',
        'Toys & Games': 'Toys & Games',
        'CDs & Vinyl': 'CDs & Vinyl',
        'Cell Phones & Accessories': 'Cell Phones & Accessories',
        'Grocery & Gourmet Food': 'Grocery & Gourmet Food',
        'Arts, Crafts & Sewing': 'Arts, Crafts & Sewing',
        'Clothing, Shoes & Jewelry': 'Clothing, Shoes & Jewelry',
        'Electronics': 'Electronics',
        'Movies & TV': 'Movies & TV',
        'Software': 'Software',
        'Video Games': 'Video Games',
        'Automotive': 'Automotive',
        'Pet Supplies': 'Pet Supplies',
        'Office Products': 'Office Products',
        'Industrial & Scientific': 'Industrial & Scientific',
        'Musical Instruments': 'Musical Instruments',
        'Tools & Home Improvement': 'Tools & Home Improvement',
        'Magazine Subscriptions': 'Magazine Subscriptions',
        'Baby Products': 'Baby Products',
        'label 25': 'label 25',
        'Appliances': 'Appliances',
        'Kitchen & Dining': 'Kitchen & Dining',
        'Collectibles & Fine Art': 'Collectibles & Fine Art',
        'All Beauty': 'All Beauty',
        'Luxury Beauty': 'Luxury Beauty',
        'Amazon Fashion': 'Amazon Fashion',
        'Computers': 'Computers',
        'All Electronics': 'All Electronics',
        'Purchase Circles': 'Purchase Circles',
        'MP3 Players & Accessories': 'MP3 Players & Accessories',
        'Gift Cards': 'Gift Cards',
        'Office & School Supplies': 'Office & School Supplies',
        'Home Improvement': 'Home Improvement',
        'Camera & Photo': 'Camera & Photo',
        'GPS & Navigation': 'GPS & Navigation',
        'Digital Music': 'Digital Music',
        'Car Electronics': 'Car Electronics',
        'Baby': 'Baby',
        'Kindle Store': 'Kindle Store',
        'Buy a Kindle': 'Buy a Kindle',
        'Furniture & D&#233;cor': 'Furniture & Decor',
        '#508510': '#508510'}

    wikics_mapping = {
        'Computational linguistics': 'Computational linguistics',
        'Databases': 'Databases',
        'Operating systems': 'Operating systems',
        'Computer architecture': 'Computer architecture',
        'Computer security': 'Computer security',
        'Internet protocols': 'Internet protocols',
        'Computer file systems': 'Computer file systems',
        'Distributed computing architecture': 'Distributed computing architecture',
        'Web technology': 'Web technology',
        'Programming language topics': 'Programming language topics'
    }

    tolokers_mapping = {
        'not banned': 'not banned',
        'banned': 'banned'
    }

    twenty_newsgroup_mapping = {'alt.atheism': 'News about atheism.', 'comp.graphics': 'News about computer graphics.', 'comp.os.ms-windows.misc': 'News about Microsoft Windows.', 'comp.sys.ibm.pc.hardware': 'News about IBM PC hardware.', 'comp.sys.mac.hardware': 'News about Mac hardware.', 'comp.windows.x': 'News about the X Window System.', 'misc.forsale': 'Items for sale.', 'rec.autos': 'News about automobiles.', 'rec.motorcycles': 'News about motorcycles.', 'rec.sport.baseball': 'News about baseball.', 'rec.sport.hockey': 'News about hockey.', 'sci.crypt': 'News about cryptography.', 'sci.electronics': 'News about electronics.', 'sci.med': 'News about medicine.', 'sci.space': 'News about space and astronomy.', 'soc.religion.christian': 'News about Christianity.', 'talk.politics.guns': 'News about gun politics.', 'talk.politics.mideast': 'News about Middle East politics.', 'talk.politics.misc': 'News about miscellaneous political topics.', 'talk.religion.misc': 'News about miscellaneous religious topics.'}



    return {
        'arxiv': arxiv_mapping, 
        'citeseer': citeseer_mapping, 
        'pubmed': pubmed_mapping, 
        'cora': cora_mapping, 
        'products': products_mapping,
        'wikics': wikics_mapping,
        'tolokers': tolokers_mapping,
        '20newsgroup': twenty_newsgroup_mapping
    }





def load_mapping():
    arxiv_mapping = {
    'arxiv cs ai': 'cs.AI',
    'arxiv cs cl': 'cs.CL',
    'arxiv cs cc': 'cs.CC',
    'arxiv cs ce': 'cs.CE',
    'arxiv cs cg': 'cs.CG',
    'arxiv cs gt': 'cs.GT',
    'arxiv cs cv': 'cs.CV',
    'arxiv cs cy': 'cs.CY',
    'arxiv cs cr': 'cs.CR',
    'arxiv cs ds': 'cs.DS',
    'arxiv cs db': 'cs.DB',
    'arxiv cs dl': 'cs.DL',
    'arxiv cs dm': 'cs.DM',
    'arxiv cs dc': 'cs.DC',
    'arxiv cs et': 'cs.ET',
    'arxiv cs fl': 'cs.FL',
    'arxiv cs gl': 'cs.GL',
    'arxiv cs gr': 'cs.GR',
    'arxiv cs ar': 'cs.AR',
    'arxiv cs hc': 'cs.HC',
    'arxiv cs ir': 'cs.IR',
    'arxiv cs it': 'cs.IT',
    'arxiv cs lo': 'cs.LO',
    'arxiv cs lg': 'cs.LG',
    'arxiv cs ms': 'cs.MS',
    'arxiv cs ma': 'cs.MA',
    'arxiv cs mm': 'cs.MM',
    'arxiv cs ni': 'cs.NI',
    'arxiv cs ne': 'cs.NE',
    'arxiv cs na': 'cs.NA',
    'arxiv cs os': 'cs.OS',
    'arxiv cs oh': 'cs.OH',
    'arxiv cs pf': 'cs.PF',
    'arxiv cs pl': 'cs.PL',
    'arxiv cs ro': 'cs.RO',
    'arxiv cs si': 'cs.SI',
    'arxiv cs se': 'cs.SE',
    'arxiv cs sd': 'cs.SD',
    'arxiv cs sc': 'cs.SC',
    'arxiv cs sy': 'cs.SY'
    }

    # arxiv_mapping = {'arxiv cs ai': 'Artificial Intelligence', 'arxiv cs cl': 'Computation and Language', 'arxiv cs cc': 'Computational Complexity', 'arxiv cs ce': 'Computational Engineering, Finance, and Science', 'arxiv cs cg': 'Computational Geometry', 'arxiv cs gt': 'Computer Science and Game Theory', 'arxiv cs cv': 'Computer Vision and Pattern Recognition', 'arxiv cs cy': 'Computers and Society', 'arxiv cs cr': 'Cryptography and Security', 'arxiv cs ds': 'Data Structures and Algorithms', 'arxiv cs db': 'Databases', 'arxiv cs dl': 'Digital Libraries', 'arxiv cs dm': 'Discrete Mathematics', 'arxiv cs dc': 'Distributed, Parallel, and Cluster Computing', 'arxiv cs et': 'Emerging Technologies', 'arxiv cs fl': 'Formal Languages and Automata Theory', 'arxiv cs gl': 'General Literature', 'arxiv cs gr': 'Graphics', 'arxiv cs ar': 'Hardware Architecture', 'arxiv cs hc': 'Human-Computer Interaction', 'arxiv cs ir': 'Information Retrieval', 'arxiv cs it': 'Information Theory', 'arxiv cs lo': 'Logic in Computer Science', 'arxiv cs lg': 'Machine Learning', 'arxiv cs ms': 'Mathematical Software', 'arxiv cs ma': 'Multiagent Systems', 'arxiv cs mm': 'Multimedia', 'arxiv cs ni': 'Networking and Internet Architecture', 'arxiv cs ne': 'Neural and Evolutionary Computing', 'arxiv cs na': 'Numerical Analysis', 'arxiv cs os': 'Operating Systems', 'arxiv cs oh': 'Other Computer Science', 'arxiv cs pf': 'Performance', 'arxiv cs pl': 'Programming Languages', 'arxiv cs ro': 'Robotics', 'arxiv cs si': 'Social and Information Networks', 'arxiv cs se': 'Software Engineering', 'arxiv cs sd': 'Sound', 'arxiv cs sc': 'Symbolic Computation', 'arxiv cs sy': 'Systems and Control'}
    citeseer_mapping = {
        "Agents": "Agents",
        "ML": "Machine Learning",
        "IR": "Information Retrieval",
        "DB": "Database",
        "HCI": "Human Computer Interaction",
        "AI": "Artificial Intelligence"
    }
    pubmed_mapping = {
        'Diabetes Mellitus, Experimental': 'Diabetes Mellitus Experimental',
        'Diabetes Mellitus Type 1': 'Diabetes Mellitus Type 1',
        'Diabetes Mellitus Type 2': 'Diabetes Mellitus Type 2'
    }
    cora_mapping = {
        'Rule_Learning': "Rule_Learning",
        'Neural_Networks': "Neural_Networks",
        'Case_Based': "Case_Based",
        'Genetic_Algorithms': "Genetic_Algorithms",
        'Theory': "Theory",
        'Reinforcement_Learning': "Reinforcement_Learning",
        'Probabilistic_Methods': "Probabilistic_Methods"
    }

    products_mapping = {'Home & Kitchen': 'Home & Kitchen',
        'Health & Personal Care': 'Health & Personal Care',
        'Beauty': 'Beauty',
        'Sports & Outdoors': 'Sports & Outdoors',
        'Books': 'Books',
        'Patio, Lawn & Garden': 'Patio, Lawn & Garden',
        'Toys & Games': 'Toys & Games',
        'CDs & Vinyl': 'CDs & Vinyl',
        'Cell Phones & Accessories': 'Cell Phones & Accessories',
        'Grocery & Gourmet Food': 'Grocery & Gourmet Food',
        'Arts, Crafts & Sewing': 'Arts, Crafts & Sewing',
        'Clothing, Shoes & Jewelry': 'Clothing, Shoes & Jewelry',
        'Electronics': 'Electronics',
        'Movies & TV': 'Movies & TV',
        'Software': 'Software',
        'Video Games': 'Video Games',
        'Automotive': 'Automotive',
        'Pet Supplies': 'Pet Supplies',
        'Office Products': 'Office Products',
        'Industrial & Scientific': 'Industrial & Scientific',
        'Musical Instruments': 'Musical Instruments',
        'Tools & Home Improvement': 'Tools & Home Improvement',
        'Magazine Subscriptions': 'Magazine Subscriptions',
        'Baby Products': 'Baby Products',
        'label 25': 'label 25',
        'Appliances': 'Appliances',
        'Kitchen & Dining': 'Kitchen & Dining',
        'Collectibles & Fine Art': 'Collectibles & Fine Art',
        'All Beauty': 'All Beauty',
        'Luxury Beauty': 'Luxury Beauty',
        'Amazon Fashion': 'Amazon Fashion',
        'Computers': 'Computers',
        'All Electronics': 'All Electronics',
        'Purchase Circles': 'Purchase Circles',
        'MP3 Players & Accessories': 'MP3 Players & Accessories',
        'Gift Cards': 'Gift Cards',
        'Office & School Supplies': 'Office & School Supplies',
        'Home Improvement': 'Home Improvement',
        'Camera & Photo': 'Camera & Photo',
        'GPS & Navigation': 'GPS & Navigation',
        'Digital Music': 'Digital Music',
        'Car Electronics': 'Car Electronics',
        'Baby': 'Baby',
        'Kindle Store': 'Kindle Store',
        'Buy a Kindle': 'Buy a Kindle',
        'Furniture & D&#233;cor': 'Furniture & Decor',
        '#508510': '#508510'}

    wikics_mapping = {
        'Computational linguistics': 'Computational linguistics',
        'Databases': 'Databases',
        'Operating systems': 'Operating systems',
        'Computer architecture': 'Computer architecture',
        'Computer security': 'Computer security',
        'Internet protocols': 'Internet protocols',
        'Computer file systems': 'Computer file systems',
        'Distributed computing architecture': 'Distributed computing architecture',
        'Web technology': 'Web technology',
        'Programming language topics': 'Programming language topics'
    }

    tolokers_mapping = {
        'not banned': 'not banned',
        'banned': 'banned'
    }
    dblp_mapping = {
        'Database': 'Database',
        'Data Mining': 'Data Mining',
        'AI': 'Artificial Intelligence',
        'Information Retrieval': 'Information Retrieval'
    }
    twenty_newsgroup_mapping = {'alt.atheism': 'News about atheism.', 'comp.graphics': 'News about computer graphics.', 'comp.os.ms-windows.misc': 'News about Microsoft Windows.', 'comp.sys.ibm.pc.hardware': 'News about IBM PC hardware.', 'comp.sys.mac.hardware': 'News about Mac hardware.', 'comp.windows.x': 'News about the X Window System.', 'misc.forsale': 'Items for sale.', 'rec.autos': 'News about automobiles.', 'rec.motorcycles': 'News about motorcycles.', 'rec.sport.baseball': 'News about baseball.', 'rec.sport.hockey': 'News about hockey.', 'sci.crypt': 'News about cryptography.', 'sci.electronics': 'News about electronics.', 'sci.med': 'News about medicine.', 'sci.space': 'News about space and astronomy.', 'soc.religion.christian': 'News about Christianity.', 'talk.politics.guns': 'News about gun politics.', 'talk.politics.mideast': 'News about Middle East politics.', 'talk.politics.misc': 'News about miscellaneous political topics.', 'talk.religion.misc': 'News about miscellaneous religious topics.'}



    return {
        'arxiv': arxiv_mapping, 
        'citeseer': citeseer_mapping, 
        'citeseer2': citeseer_mapping,
        'pubmed': pubmed_mapping, 
        'cora': cora_mapping, 
        'products': products_mapping,
        'wikics': wikics_mapping,
        'tolokers': tolokers_mapping,
        '20newsgroup': twenty_newsgroup_mapping,
        'dblp': dblp_mapping,
    }

def retrieve_dict(clean_t):
    start = clean_t.find('[')
    end = clean_t.find(']', start) + 1  # +1 to include the closing bracket
    list_str = clean_t[start:end]
    result = ast.literal_eval(list_str)
    return result


def compute_ece(confidences, predictions, labels, n_bins=10):
    """
    Compute Expected Calibration Error.
    
    Parameters:
    - confidences (Tensor): Tensor of predicted confidences
    - predictions (Tensor): Tensor of predicted classes
    - labels (Tensor): Tensor of true labels
    - n_bins (int): Number of bins to use for calibration

    Returns:
    - ece (float): Expected Calibration Error
    """
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower.item()) * (confidences <= bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin * ((bin_lower + bin_upper) / 2.0)

    return ece

# Example usage:
# confidences = torch.tensor([...])  # Replace with your confidence values
# predictions = torch.tensor([...])  # Replace with your predicted class labels
# labels = torch.tensor([...])  # Replace with your true labels

# ece = compute_ece(confidences, predictions, labels, n_bins=10)
# print(f"Expected Calibration Error: {ece:.2%}")




def plot_calibration_curve(confidences, predictions, labels, data_name = 'cora', method = 'zero_shot', n_bins=10):
    """
    Plots the calibration curve.
    
    Parameters:
    - confidences (Tensor): Tensor of predicted confidences.
    - predictions (Tensor): Tensor of predicted classes.
    - labels (Tensor): Tensor of true labels.
    - n_bins (int): Number of bins to use for calibration.
    """
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_confidences = []
    bin_accuracies = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
        if in_bin.sum().item() > 0:
            bin_confidence = confidences[in_bin].mean().item()
            bin_accuracy = (predictions[in_bin] == labels[in_bin]).float().mean().item()

            # bin_confidences.append(bin_confidence)
            bin_confidences.append(bin_lower.item())
            bin_accuracies.append(bin_accuracy)

    # Plotting
    plt.bar(bin_confidences, bin_accuracies, width=0.01, edgecolor="black", alpha=0.6, label="Model Accuracy")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Perfect Calibration")
    plt.xlabel("Average Confidence of Bins")
    plt.ylabel("Accuracy")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig("{}_{}_calibration_curve.png".format(data_name, method))
    plt.clf()

# Example usage:
# confidences = torch.tensor([...])  # Replace with your confidence values
# predictions = torch.tensor([...])  # Replace with your predicted class labels
# labels = torch.tensor([...])  # Replace with your true labels

# plot_calibration_curve(confidences, predictions, labels, n_bins=10)

def pair_wise_prediction(configs, dataset,dataname, node_idx1, node_idx2, ensemble_ca_matrix):
    # object_cat = configs[dataname]['pair']['object-cat']
    question = configs[dataname]['pair']['question']
    answer_format = configs[dataname]['pair']['answer-format']
    try:
        if dataname in ['pubmed']:
            dataset.label_names[0] = dataset.label_names[0].replace(",", "")
            prompt = cluster_zero_shot_prompt(dataset.raw_texts[node_idx1], dataset.raw_texts[node_idx2], label_names = dataset.label_names, need_tasks = True, object_cat = "", question = question, answer_format = answer_format)
        else:
            prompt = cluster_zero_shot_prompt(dataset.raw_text[node_idx1], dataset.raw_text[node_idx2], label_names = dataset.label_names, need_tasks = True, object_cat = "", question = question, answer_format = answer_format)
    except:
        import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    system_prompt = "You are a model that especially good at clustering papers."
    import openai
    # openai.api_base = "https://api.openai-proxy.org/v1"
    openai.api_base = "change to your endpoint"
    openai.api_key = "change to your key"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt,
                "role": "user", "content": prompt,},
        ]
    )
    try:
        res = json.loads(response['choices'][0]['message']['content'])
        answer = res[0]['answer']
        confidence = res[0]['confidence']
    except:
        return -1, -1
    if answer == "No":
        ensemble_ca_matrix[node_idx1, node_idx2] = 0
        answer = 0
    elif answer == "Yes":
        ensemble_ca_matrix[node_idx1, node_idx2] = 1
        answer = 1
    else:
        return -1, -1
    return answer, confidence


def cluster_zero_shot_prompt(node1_text, node2_text, label_names, need_tasks = True, object_cat = "Paper", question = "Which arxiv cs subcategories does this node belong to?", \
                    answer_format = "Give 3 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely together with a confidence ranging from 0 to 100, in the form of a list python dicts like [{\"answer:\":<answer_here>, \"confidence\": <confidence_here>}]"):
    prompt = "{}: \n".format(object_cat)
    prompt += "Paper 1:" + (node1_text + "\n")
    prompt += "Paper 2:" + (node2_text + "\n")
    if not 'arxiv' in question:
        prompt += "There are following categories: \n"
        prompt += "[" + ", ".join(label_names) + "]" + "\n"
    prompt += "Task: \n"
    prompt += question + "?\n"
    if need_tasks:
        prompt +=  answer_format
   
    return prompt

def cluster_few_shot_prompt(dataset_prompt, node1_text, flag_texts, label_names, need_tasks = True, object_cat = "Paper", question = "Which arxiv cs subcategories does this node belong to?", \
                    answer_format = "Give 3 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely together with a confidence ranging from 0 to 100, in the form of a list python dicts like [{\"answer:\":<answer_here>, \"confidence\": <confidence_here>}]",
                    reasoning = "", dataname='cora'):
    prompts = []
    prompt = dataset_prompt
    if not 'arxiv' in question:
        prompt += "All possible categories: \n"
        prompt += "[" + ", ".join(label_names) + "]" + "\n"
        if dataname != 'dblp':
            prompt += reasoning + "\n"
    # for idx, text in enumerate(flag_texts):
    #     prompt += "Example {}: ".format(idx) + (text + "\n")
    #     import random
    #     prompt += "answer: <{}>, confidence: <{}>".format(label_names[idx], random.choice([100,98,95])) + "\n"
    prompt += "Target paper:" + (node1_text + "\n")
    prompt+= answer_format + "\n"
    return prompt

def cluster_similarity_prompt(node1_text, node2_text, node2_category, label_names, need_tasks = True, object_cat = "Paper", question = "Which arxiv cs subcategories does this node belong to?", \
                    answer_format = "Give 3 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely together with a confidence ranging from 0 to 100, in the form of a list python dicts like [{\"answer:\":<answer_here>, \"confidence\": <confidence_here>}]",
                    reasoning = ""):
    prompt = "You are a model that especially good at classifying paper's category. Now I will first give you all the possible categories and their explanation. Then I will give you a source paper and it's category. Please answer the following question: "+ question+"? \n"
    if not 'arxiv' in question:
        prompt += "All possible categories: \n"
        prompt += "[" + ", ".join(label_names) + "]" + "\n"
        prompt += reasoning + "\n"
    prompt += "Source paper: " + (node2_text + "\n")
    prompt += "Source paper category: " + (node2_category + "\n")
    prompt += "Target paper:" + (node1_text + "\n")
    return prompt

def get_example_prediction(data, pred, dataset='cora', save_path=""):
    '''
    For each category, get one most confident prediction.
    return the index and the prediction 
    '''
    import os
    # import pdb; pdb.set_trace()
    res_idx = [-1]*len(data.label_names)
    res_label = [-1]*len(data.label_names)
    output_dict = {}
    valid_preds_idx = torch.where(pred['pred']>=0)[0]
    filter_texts = []
    filter_y = []
    for idx in valid_preds_idx:
        filter_y.append(data.y[idx.item()])
        filter_texts.append(data.raw_texts[idx.item()])
    valid_preds = pred['pred'][valid_preds_idx]
    valid_confs = pred['conf'][valid_preds_idx]
    valid_preds_idx = torch.argsort(valid_confs, descending = True)
    for idx in valid_preds_idx:
        idx = idx.item()
        if min(res_label) > -1:
            for out_idx, answer in zip(res_idx, res_label):
                output_dict[answer.item()] = filter_texts[out_idx]
                print(f"LLM Category: {answer.item()}, Real Category: {filter_y[out_idx]}")
            save_path = os.path.join(save_path, "confident_predictions_{}.json".format(dataset))
            with open(save_path, 'w') as f:
                json.dump(output_dict, f)
            return
        else:
            # import pdb; pdb.set_trace()
            if res_idx[valid_preds[idx]] == -1 and len(filter_texts[idx]) <512:
                res_idx[valid_preds[idx]] = idx
                res_label[valid_preds[idx]] = valid_preds[idx]
    print("No enough confident predictions")