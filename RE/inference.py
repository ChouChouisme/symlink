import os
import torch.nn.functional as F
import torch
import json
import argparse
from collections import Counter
from util.dataset import LoadDataset
from torch.utils.data import DataLoader
from util.model import RE_classifier
import tqdm
label_map = ['Count', 'Direct', 'Corefer-Symbol', 'Corefer-Description', 'Negative_Sample']

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="inference the model output.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--file_name", type=str, required=True)

    return parser

def evaluation(best_model_path, corpus_path, seq_len=512, batch_size=200, device='cuda:0',num_workers=5, model_name='base'
               , index_to_docid='', all_data=[]):

    test_dataset = LoadDataset(corpus_path=corpus_path, seq_len=seq_len, mode='infer', model_name=model_name)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    model = RE_classifier(resize_token_embd_len=test_dataset.get_tokenizer_len(), model_name=model_name)
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    
    model.to(device)
    model.eval()
    predict_list = []
    result_dict = {}
    
    with torch.no_grad():
        for index, data in tqdm.tqdm(enumerate(test_data_loader)):
            data = {key: value.to(device) for key, value in data.items()}
            # logits = model.forward(**data)
            logits = model.forward(input_ids=data['input_ids'], attention_mask=data['attention_mask'],  span_1=data['span_1'], span_2=data['span_2'])
            
            predict = F.softmax(logits, dim=1).argmax(dim=1)
            predict_list += predict.tolist()

            predicted_doc_num = data['doc_num'].tolist()
            predicted_e1_id = data['e1_id'].tolist()
            predicted_e2_id = data['e2_id'].tolist()
            predicted_label = predict.tolist()
            # print(predict.tolist())
            
                   
            def check_valid_rel(label, arg0, arg1, all_data, doc_id):
                arg0_label = all_data[doc_id]['entity'][arg0]['label']
                arg1_label = all_data[doc_id]['entity'][arg1]['label']
                
                
                if label == 'Count' or label == 'Direct':
                    if arg0_label == arg1_label:
                        return False
                    else:
                        if arg0_label == 'SYMBOL':
                            return 2
                        else:
                            return True
                elif label == 'Corefer-Symbol':
                    if arg0_label == arg1_label and arg0_label == 'SYMBOL':
                        return True
                    else:
                        return False
                elif label == 'Corefer-Description':
                    if arg0_label == arg1_label and arg0_label == 'PRIMARY':
                        return True
                    else:
                        return False
                elif label == 'Negative_Sample':
                    return False            

                            
            for i in range(len(predicted_label)):
                # print(result_dict)
                
                e1 = 'T'+str(predicted_e1_id[i])
                e2 = 'T'+str(predicted_e2_id[i])
                label = label_map[predicted_label[i]]
                
                if predicted_doc_num[i] not in result_dict:
                    result_dict[predicted_doc_num[i]] = []
                    check_valid = check_valid_rel(label, e1, e2, all_data, index_to_docid[predicted_doc_num[i]])
                    if check_valid == 2: # change
                        if e1 != e2:
                            e1, e2 = e2, e1
                        result_dict[predicted_doc_num[i]].append({"label": label, "arg0": e1, "arg1": e2})
                    elif check_valid == 1: # True
                        result_dict[predicted_doc_num[i]].append({"label": label, "arg0": e1, "arg1": e2})                        
                else:
                    if len(result_dict[predicted_doc_num[i]]) >= 1:
                        flag = True
                        for rel in result_dict[predicted_doc_num[i]]:
                            if (rel['arg0'] == e1 and rel['arg1'] == e2) or (rel['arg1'] == e1 and rel['arg0'] == e2):
                                flag = False
                                break
                        if flag == False:
                            continue
                            
                        check_valid = check_valid_rel(label, e1, e2, all_data, index_to_docid[predicted_doc_num[i]])
                        if check_valid == 2: # change
                            if e1 != e2:
                                e1, e2 = e2, e1
                            result_dict[predicted_doc_num[i]].append({"label": label, "arg0": e1, "arg1": e2})
                        elif check_valid == 1: # True
                            result_dict[predicted_doc_num[i]].append({"label": label, "arg0": e1, "arg1": e2})                                
                    else:
                        check_valid = check_valid_rel(label, e1, e2, all_data, index_to_docid[predicted_doc_num[i]])
                        if check_valid == 2: # change
                            if e1 != e2:
                                e1, e2 = e2, e1
                            result_dict[predicted_doc_num[i]].append({"label": label, "arg0": e1, "arg1": e2})
                        elif check_valid == 1: # True
                            result_dict[predicted_doc_num[i]].append({"label": label, "arg0": e1, "arg1": e2})        
    return result_dict

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    device = 'cuda:4'
    best_model_path = '/data0/wxl/symlink/model/re/20400_95.44_332_scibert_uncased'
    corpus_path = args.data_dir
    result_path = args.output_dir
    file_name = args.file_name
    model_name = 'scibert_uncased'
    seq_len = 512
    batch_size = 200
    
    ##################
    
    all_data = json.load(open(corpus_path, encoding="utf-8"))
    
    index_to_docid = {}
    
    for index, doc_id in enumerate(all_data):
        index_to_docid[index] = doc_id
    
    result = evaluation(best_model_path= best_model_path, corpus_path=corpus_path,
                                         seq_len=seq_len, batch_size=batch_size, device=device, num_workers=5,
                                         model_name=model_name, index_to_docid=index_to_docid, all_data=all_data)
    for id, doc_id in enumerate(all_data):
        all_data[doc_id]['relation'] = {}
        
        if id not in result:
            continue
        
        for rid, relation in enumerate(result[id]):
            relation['rid'] = 'R'+str(rid+1)
            all_data[doc_id]['relation']['R'+str(rid+1)] = relation

    def exclude_not_related_entities(result):
        result_temp = result
        
        for doc_name in result_temp:
            rels = result_temp[doc_name]['relation']
            used_entities = []
            for rel_name in result[doc_name]['relation']:
                used_entities.append(rels[rel_name]['arg0'])
                used_entities.append(rels[rel_name]['arg1'])
            
            entities_name_li = []
            
            for ent_name in result_temp[doc_name]['entity']:
                entities_name_li.append(ent_name)
            
            for ent_name in entities_name_li:
                if ent_name not in used_entities:
                    del result[doc_name]['entity'][ent_name]                                   
        return result
    
    def postpro_corefer_relations(result):
        for doc_name in result:
            rel_names = list(result[doc_name]['relation'].keys())
            rels = result[doc_name]['relation']
            
            corefer_sym_li = []
            corefer_desc_li = []
            
            for rel_name in rel_names:
                arg0 = rels[rel_name]['arg0']
                arg1 = rels[rel_name]['arg1']
                
                if rels[rel_name]['label'] == 'Corefer-Symbol':
                    del result[doc_name]['relation'][rel_name]                 

                    temp_flag = False
                    for i, corefer_sym in enumerate(corefer_sym_li):
                        if arg0 in corefer_sym:
                            if arg1 not in corefer_sym:
                                corefer_sym_li[i].append(arg1)
                                
                            temp_flag = True
                            break
                        elif arg1 in corefer_sym:
                            corefer_sym_li[i].append(arg0)
                            temp_flag = True
                            break
                    if temp_flag:
                        continue
                    else:
                        corefer_sym_li.append([arg0, arg1])
                elif rels[rel_name]['label'] == 'Corefer-Description':
                    del result[doc_name]['relation'][rel_name]                 

                    temp_flag = False
                    for i, corefer_desc in enumerate(corefer_desc_li):
                        if arg0 in corefer_desc:
                            if arg1 not in corefer_desc:
                                corefer_desc_li[i].append(arg1)
                                
                            temp_flag = True
                            break
                        elif arg1 in corefer_desc:
                            corefer_desc_li[i].append(arg0)
                            temp_flag = True
                            break
                    if temp_flag:
                        continue
                    else:
                        corefer_desc_li.append([arg0, arg1])
                        
            temp_counter = 0
            for corefer_syms in corefer_sym_li:
                iter_num = len(corefer_syms) - 1
                for i in range(iter_num):
                    result[doc_name]['relation']['R50'+str(temp_counter)] = \
                        {"label": "Corefer-Symbol", "arg0": corefer_syms[i], "arg1": corefer_syms[i+1], "rid": 'R50'+str(temp_counter)}
                    temp_counter += 1
            for corefer_desc in corefer_desc_li:
                iter_num = len(corefer_desc) - 1
                for i in range(iter_num):
                    result[doc_name]['relation']['R50'+str(temp_counter)] = \
                        {"label": "Corefer-Description", "arg0": corefer_desc[i], "arg1": corefer_desc[i+1], "rid": 'R50'+str(temp_counter)}
                    temp_counter += 1            
                              
        return result
    
    all_data = exclude_not_related_entities(all_data)
    all_data = postpro_corefer_relations(all_data)
    
    answers = set()
    for doc_name in all_data:
        ents = all_data[doc_name]['entity']
        rels = all_data[doc_name]['relation']
        for rel_name in all_data[doc_name]['relation']:
            e1id = rels[rel_name]['arg0']
            e2id = rels[rel_name]['arg1']
            rel = {
                "label1": ents[e1id]['label'],
                "text1": ents[e1id]['text'],
                "label2": ents[e2id]['label'],
                "text2": ents[e2id]['text'],
                "label": rels[rel_name]['label']
            }
            answers.add(tuple(rel.items()))
    
    entities = {}
    relations = []

    for ans in answers:
        answer = dict(ans)
        ent1 = {
            "label": answer['label1'],
            "text": answer['text1']
        }
        ent2 = {
            "label": answer['label2'],
            "text": answer['text2']
        }
        
        if ent1 == ent2:
            continue
        
        if ent1 not in entities.values():
            eid1 = f"T{len(entities)+1}"
            entities[eid1] = ent1
        else:
            eid1 = next(eid for eid, entity in entities.items() if entity == ent1)

        if ent2 not in entities.values():
            eid2 = f"T{len(entities)+1}"
            entities[eid2] = ent2
        else:
            eid2 = next(eid for eid, entity in entities.items() if entity == ent2)

        answer['arg0'] = eid1
        answer['arg1'] = eid2
        relations.append(answer)
    
    with open(os.path.join(result_path, f"{file_name}.json"), 'w', encoding='utf-8') as make_file:
        json.dump({"entity": entities, "relation": relations}, make_file, indent="\t")
    
    corefer_sym_li = []
    corefer_desc_li = []
    
    for rel in relations:
        arg0 = rel['arg0']
        arg1 = rel['arg1']
        
        if rel['label'] == 'Corefer-Symbol':
            temp_flag = False
            for i, corefer_sym in enumerate(corefer_sym_li):
                if arg0 in corefer_sym:
                    if arg1 not in corefer_sym:
                        corefer_sym_li[i].append(arg1)
                        
                    temp_flag = True
                    break
                elif arg1 in corefer_sym:
                    corefer_sym_li[i].append(arg0)
                    temp_flag = True
                    break
            if temp_flag:
                continue
            else:
                corefer_sym_li.append([arg0, arg1])
        elif rel['label'] == 'Corefer-Description':
            temp_flag = False
            for i, corefer_desc in enumerate(corefer_desc_li):
                if arg0 in corefer_desc:
                    if arg1 not in corefer_desc:
                        corefer_desc_li[i].append(arg1)
                        
                    temp_flag = True
                    break
                elif arg1 in corefer_desc:
                    corefer_desc_li[i].append(arg0)
                    temp_flag = True
                    break
            if temp_flag:
                continue
            else:
                corefer_desc_li.append([arg0, arg1])

    symbol_dict = {}
    for rel in relations:
        if rel["label"]=="Direct":
            arg0 = rel["arg0"]
            arg1 = rel["arg1"]
            now_corefer_desc = [arg0]
            now_corefer_sym = [arg1]
            for corefer_desc in corefer_desc_li:
                if arg0 in corefer_desc:
                    now_corefer_desc = corefer_desc
            for corefer_sym in corefer_sym_li:
                if arg1 in corefer_sym:
                    now_corefer_sym = corefer_sym
            ndesc = [entities[ent]["text"] for ent in now_corefer_desc]
            nsym = [entities[ent]["text"] for ent in now_corefer_sym]
            for sym in nsym:
                if sym not in symbol_dict:
                    symbol_dict[sym] = ndesc
                else:
                    symbol_dict[sym] = list(set(symbol_dict[sym] + ndesc))

    with open(os.path.join(result_path, f"{file_name}_dict.json"), 'w', encoding='utf-8') as make_file:
        json.dump(symbol_dict, make_file, indent="\t")